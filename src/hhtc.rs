use blake3::hash;
use flate2::{write::GzEncoder, Compression};
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::io::Write;
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

use crate::embedder::TinyBertEmbedder;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PrecompState {
    pub compressed_kv: Vec<u8>,
    pub embedding: Vec<f32>,
}

pub struct HHTCEngine {
    chunk_size: usize,
    cache: Mutex<LruCache<u64, PrecompState>>,
    embedder: Arc<Mutex<TinyBertEmbedder>>,
}

impl HHTCEngine {
    pub async fn new(chunk_size: usize, cache_capacity: usize) -> Result<Self, Box<dyn Error>> {
        let embedder = Arc::new(Mutex::new(TinyBertEmbedder::new(768).await?));
        Ok(Self {
            chunk_size,
            cache: Mutex::new(LruCache::new(NonZeroUsize::new(cache_capacity).unwrap())),
            embedder,
        })
    }

    pub async fn compress(&mut self, text: &str) -> (String, f64) {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let original_tokens_count = tokens.len();

        if original_tokens_count == 0 {
            return (String::new(), 1.0);
        }

        let mut compressed_output_string = String::new();
        let mut actual_compressed_token_count = 0;

        let mut current_token_idx = 0;
        while current_token_idx < original_tokens_count {
            let remaining_tokens = original_tokens_count - current_token_idx;
            let chunk_size_for_current_segment = std::cmp::min(self.chunk_size, remaining_tokens);

            let chunk_tokens = &tokens[current_token_idx..current_token_idx + chunk_size_for_current_segment];
            let chunk_text = chunk_tokens.join(" ");

            let chunk_hash_bytes = hash(chunk_text.as_bytes()).as_bytes()[0..8].try_into().unwrap();
            let hash_id = u64::from_le_bytes(chunk_hash_bytes);

            let mut lru_cache = self.cache.lock().await;
            if lru_cache.contains(&hash_id) {
                compressed_output_string.push_str(&format!("#{} ", hash_id));
                actual_compressed_token_count += 1;
            } else {
                let embedder_locked = self.embedder.lock().await;
                let embedding = embedder_locked.embed(&chunk_text).await.unwrap_or_else(|e| {
                    info!("Error computing embedding for HHTC chunk: {:?}", e);
                    vec![0.0; embedder_locked.dim()]
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

        let compression_ratio = actual_compressed_token_count as f64 / original_tokens_count as f64;

        (compressed_output_string.trim().to_string(), compression_ratio)
    }
}

