use rocksdb::{DB, Options};
use std::path::Path;
use serde::{Serialize, Deserialize};
use tracing::info;
use candle_core::{Tensor, Device, DType};
use candle_nn::{Linear, VarBuilder, Module, linear};
use hf_hub::{api::tokio::Api as AsyncApi, Repo, RepoType};
use tokenizers::Tokenizer;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::error::Error;
use crate::hhtc::TinyBertEmbedder;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBlob {
    pub key: String,
    pub value: String,
    pub confidence: f64,
    pub embedding: Vec<f32>,
}

pub struct EKFStorage {
    db: DB,
    embedder: Arc<Mutex<TinyBertEmbedder>>,
    vector_index: Arc<Mutex<Vec<(String, Vec<f32>)>>>,
}

impl EKFStorage {
    pub async fn new(path: &Path, tokenizer: Arc<Mutex<Tokenizer>>, device: Arc<Device>) -> Result<Self, Box<dyn Error>> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        let db = DB::open(&opts, path)?;
        
        info!("EKF storage initialized at: {:?}", path);
        
        let embedder = Arc::new(Mutex::new(TinyBertEmbedder::new(30522, 768, device, tokenizer).await?));
        
        let vector_index = Self::populate_initial_knowledge(&db, embedder.clone()).await?;

        Ok(Self { db, embedder, vector_index: Arc::new(Mutex::new(vector_index)) })
    }

    async fn populate_initial_knowledge(db: &DB, embedder: Arc<Mutex<TinyBertEmbedder>>) -> Result<Vec<(String, Vec<f32>)>, Box<dyn Error>> {
        let initial_facts_data = vec![
            ("transformer", "Transformers are a deep learning architecture that relies on self-attention mechanisms."),
            ("attention", "Attention mechanisms allow a model to weigh the importance of different parts of the input sequence."),
            ("Rust", "Rust is a systems programming language focused on safety, speed, and concurrency."),
            ("unsupervised learning", "Unsupervised learning is a type of machine learning that looks for patterns in data without explicit labels."),
            ("supervised learning", "Supervised learning uses labeled data to train a model."),
            ("garbage collection", "Rust is not a garbage collected language and manages memory ownership at compile time."),
            ("bottleneck", "LLM inference is often bottlenecked by memory bandwidth, not just compute."),
            ("France capital", "The capital of France is Paris."),
            ("Paris", "Paris is known for its Eiffel Tower, Louvre Museum, and rich history."),
            ("car speed", "A bicycle is not the fastest car in the world."),
            ("fastest car", "The world's fastest cars are designed for high speeds, not cycling.")
        ];
        
        let mut vector_index = Vec::new();
        let embedder_locked = embedder.lock().await;

        for (key_str, value_str) in initial_facts_data {
            let embedding = embedder_locked.embed(value_str).await?;
            let fact = KnowledgeBlob { 
                key: key_str.to_string(), 
                value: value_str.to_string(), 
                confidence: 0.95,
                embedding,
            };
            let serialized = serde_json::to_vec(&fact).unwrap();
            db.put(key_str, serialized).unwrap();
            vector_index.push((fact.key.clone(), fact.embedding));
        }
        
        info!("EKF pre-populated with initial knowledge.");
        Ok(vector_index)
    }

    pub async fn query(&mut self, prompt: &str) -> Result<Vec<String>, Box<dyn Error>> {
        let embedder_locked = self.embedder.lock().await;
        let prompt_embedding = embedder_locked.embed(prompt).await?;
        
        let vector_index_locked = self.vector_index.lock().await;
        let mut similarities = Vec::new();

        for (key, embedding) in vector_index_locked.iter() {
            let similarity = Self::cosine_similarity(&prompt_embedding, embedding);
            similarities.push((key.clone(), similarity));
        }
        
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(3);
        
        let mut results = Vec::new();
        for (key, sim) in similarities {
            if sim > 0.6 {
                if let Ok(Some(value_bytes)) = self.db.get(&key) {
                    if let Ok(blob) = serde_json::from_slice::<KnowledgeBlob>(&value_bytes) {
                        results.push(blob.value);
                    }
                }
            }
        }
        Ok(results)
    }
    
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        
        for i in 0..a.len() {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            0.0
        } else {
            dot_product / denom
        }
    }
}