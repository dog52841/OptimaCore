use blake3::hash;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::error::Error;

/// A lightweight, deterministic embedder used by both the HHTC engine and
/// EKF storage. It avoids external model downloads by generating a
/// pseudo-random embedding based on the BLAKE3 hash of the input text.
pub struct TinyBertEmbedder {
    dim: usize,
}

impl TinyBertEmbedder {
    /// Create a new embedder with the given dimensionality.
    pub async fn new(dim: usize) -> Result<Self, Box<dyn Error>> {
        Ok(Self { dim })
    }

    /// Generate a deterministic embedding for `text`.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        let mut embedding = vec![0f32; self.dim];
        let mut rng = StdRng::from_seed(hash(text.as_bytes()).as_bytes()[..32].try_into().unwrap());
        for v in embedding.iter_mut() {
            *v = rng.gen::<f32>();
        }
        Ok(embedding)
    }

    /// Return the embedding dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

