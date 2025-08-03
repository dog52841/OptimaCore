use blake3::hash;
use dashmap::DashMap;
use lru::LruCache;
use std::sync::Mutex;
use std::num::NonZeroUsize;
use serde::{Serialize, Deserialize};
use flate2::{write::GzEncoder, Compression};
use std::io::Write;
use tokenizers::Tokenizer;
use candle_core::{Tensor, Device, DType};
use candle_nn::{Linear, VarBuilder, Module, linear};
use hf_hub::{api::tokio::Api as AsyncApi, Repo, RepoType};
use tracing::info;
use std::sync::Arc;
use std::error::Error;
use crate::ekf::TinyBertEmbedder;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PrecompState {
    pub compressed_kv: Vec<u8>,
    pub embedding: Vec<f32>,
}

pub struct HHTCEngine {
    chunk_size: usize,
    cache: Mutex<LruCache<u64, PrecompState>>,
    tokenizer: Arc<Mutex<Tokenizer>>,
    embedder: Arc<Mutex<TinyBertEmbedder>>,
}

impl HHTCEngine {
    pub async fn new(chunk_size: usize, cache_capacity: usize, device: Arc<Device>) -> Result<Self, Box<dyn Error>> {
        let api = AsyncApi::new()?;
        let tokenizer_path = api.repo(Repo::with_revision(
            "bert-base-uncased".to_string(),
            RepoType::Model,
            "main".to_string(),
        )).get("tokenizer.json").await?;
        let tokenizer = Arc::new(Mutex::new(Tokenizer::from_file(tokenizer_path)?));

        let embedder = Arc::new(Mutex::new(TinyBertEmbedder::new(30522, 768, device, tokenizer.clone()).await?));

        Ok(Self {
            chunk_size,
            cache: Mutex::new(LruCache::new(NonZeroUsize::new(cache_capacity).unwrap())),
            tokenizer,
            embedder,
        })
    }
    
    pub fn get_tokenizer_arc(&self) -> Arc<Mutex<Tokenizer>> {
        self.tokenizer.clone()
    }
    
    pub async fn compress(&mut self, text: &str) -> (String, f64) {
        let tokenizer_locked = self.tokenizer.lock().await;
        let encoded_tokens = match tokenizer_locked.encode(text, true) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(e) => {
                info!("Tokenizer encoding error: {:?}. Falling back to basic split.", e);
                text.split_whitespace().map(|_| 0u32).collect()
            }
        };
        let original_tokens_count = encoded_tokens.len();

        let mut compressed_output_string = String::new();
        let mut actual_compressed_token_count = 0;

        if original_tokens_count == 0 {
            return (String::new(), 1.0);
        }

        let mut current_token_idx = 0;
        while current_token_idx < original_tokens_count {
            let remaining_tokens = original_tokens_count - current_token_idx;
            let chunk_size_for_current_segment = std::cmp::min(self.chunk_size, remaining_tokens);
            
            let chunk_token_ids_slice = &encoded_tokens[current_token_idx..current_token_idx + chunk_size_for_current_segment];
            let chunk_text = tokenizer_locked.decode(chunk_token_ids_slice, true).unwrap_or_default();

            let chunk_hash_bytes = hash(chunk_text.as_bytes()).as_bytes()[0..8].try_into().unwrap();
            let hash_id = u64::from_le_bytes(chunk_hash_bytes);

            let mut lru_cache = self.cache.lock().unwrap();
            if lru_cache.contains(&hash_id) {
                compressed_output_string.push_str(&format!("#{} ", hash_id));
                actual_compressed_token_count += 1;
            } else {
                let embedder_locked = self.embedder.lock().await;
                let embedding = embedder_locked.embed(&chunk_text).await.unwrap_or_else(|e| {
                    info!("Error computing embedding for HHTC chunk: {:?}", e);
                    vec![0.0; 768]
                });
                
                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(chunk_text.as_bytes()).unwrap();
                let compressed_kv_data = encoder.finish().unwrap();
                
                let precomp_state = PrecompState {
                    compressed_kv: compressed_kv_data,
                    embedding,
                };
                
                lru_cache.put(hash_id, precomp_state);
                compressed_output_string.push_str(&chunk_text);
                compressed_output_string.push(' ');
                actual_compressed_token_count += chunk_size_for_current_segment;
            }
            current_token_idx += chunk_size_for_current_segment;
        }

        let compression_ratio = if original_tokens_count > 0 {
            actual_compressed_token_count as f64 / original_tokens_count as f64
        } else {
            1.0
        };

        (compressed_output_string.trim().to_string(), compression_ratio)
    }
}