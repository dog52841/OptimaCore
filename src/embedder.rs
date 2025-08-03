use std::error::Error;
use std::sync::Arc;
use tokio::task;
use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType};

/// A real sentence embedder backed by a pre-trained transformer model
/// downloaded from Hugging Face. It produces deterministic embeddings
/// for arbitrary input text.
pub struct SentenceEmbedder {
    model: Arc<SentenceEmbeddingsModel>,
    dim: usize,
}

impl SentenceEmbedder {
    /// Initialize the embedder using the `all-MiniLM-L6-v2` model.
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        let model = task::spawn_blocking(|| {
            SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
                .create_model()
        })
        .await??;
        // Determine dimensionality
        let dim = model.encode(&["dim"])?[0].len();
        Ok(Self { model: Arc::new(model), dim })
    }

    /// Generate an embedding for `text`.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        let model = self.model.clone();
        let text = text.to_string();
        let embedding = task::spawn_blocking(move || model.encode(&[text])).await??;
        Ok(embedding[0].clone())
    }

    /// Return the embedding dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }
}
