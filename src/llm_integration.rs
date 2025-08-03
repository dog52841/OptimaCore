use std::error::Error;
use std::sync::{Arc, Mutex};
use tokio::task;
use rust_bert::pipelines::generation::{GPT2Generator, GenerateConfig, LanguageGenerator};
use tracing::info;

/// Client providing local GPT-2 text generation.
pub struct LLMClient {
    generator: Arc<Mutex<GPT2Generator>>, 
}

impl LLMClient {
    /// Initialize the GPT-2 generator with default settings.
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        let generator = task::spawn_blocking(|| {
            let config = GenerateConfig::default();
            GPT2Generator::new(config)
        })
        .await??;
        info!("GPT-2 generator initialized");
        Ok(Self { generator: Arc::new(Mutex::new(generator)) })
    }

    /// Generate text for the provided prompt and optional context.
    pub async fn generate(&self, prompt: &str, context: &[String]) -> Result<String, Box<dyn Error>> {
        let context_str = if context.is_empty() {
            String::new()
        } else {
            format!("Context from Knowledge Folder: {}\n\n", context.join("\n"))
        };
        let full_prompt = format!("{}{}", context_str, prompt);
        let generator = self.generator.clone();
        let input = vec![full_prompt];
        let output = task::spawn_blocking(move || {
            let mut gen = generator.lock().unwrap();
            gen.generate(Some(&input), None)
        })
        .await??;
        Ok(output[0].clone())
    }
}
