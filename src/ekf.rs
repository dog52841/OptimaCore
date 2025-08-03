use rocksdb::{DB, Options, IteratorMode};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

use crate::embedder::SentenceEmbedder;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBlob {
    pub key: String,
    pub value: String,
    pub confidence: f64,
    pub embedding: Vec<f32>,
}

pub struct EKFStorage {
    db: DB,
    embedder: Arc<Mutex<SentenceEmbedder>>,
    vector_index: Arc<Mutex<Vec<(String, Vec<f32>)>>>,
}

impl EKFStorage {
    pub async fn new(path: &Path) -> Result<Self, Box<dyn Error>> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        let db = DB::open(&opts, path)?;

        info!("EKF storage initialized at: {:?}", path);

        let embedder = Arc::new(Mutex::new(SentenceEmbedder::new().await?));
        let vector_index = Self::load_index(&db).await?;

        Ok(Self { db, embedder, vector_index: Arc::new(Mutex::new(vector_index)) })
    }

    async fn load_index(db: &DB) -> Result<Vec<(String, Vec<f32>)>, Box<dyn Error>> {
        let mut vector_index = Vec::new();
        let iter = db.iterator(IteratorMode::Start);
        for item in iter {
            let (key, value) = item?;
            if let Ok(blob) = serde_json::from_slice::<KnowledgeBlob>(&value) {
                vector_index.push((String::from_utf8(key.to_vec())?, blob.embedding));
            }
        }
        Ok(vector_index)
    }

    pub async fn query(&self, prompt: &str) -> Result<Vec<String>, Box<dyn Error>> {
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

