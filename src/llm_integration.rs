use serde_json::json;
use reqwest::Client;
use tracing::info;
use std::error::Error;

pub struct LLMClient {
    client: Client,
    api_endpoint: String,
}

impl LLMClient {
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        let client = Client::new();
        let api_endpoint = std::env::var("LLM_API_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:8000/generate".to_string());
        
        info!("LLMClient initialized. Target API: {}", api_endpoint);
        Ok(Self { client, api_endpoint })
    }
    
    pub async fn generate(&self, prompt: &str, context: &[String]) -> Result<String, Box<dyn Error>> {
        let context_str = if context.is_empty() {
            String::new()
        } else {
            format!("Context from Knowledge Folder: {}\n\n", context.join("\n"))
        };
        
        let full_prompt = format!("{}{}", context_str, prompt);
        
        let payload = json!({
            "prompt": full_prompt,
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
        });
        
        info!("Sending request to LLM API: {}", self.api_endpoint);
        
        let response = self.client
            .post(&self.api_endpoint)
            .json(&payload)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("LLM API request failed: {} - {}", response.status(), error_text).into());
        }
        
        let response_json: serde_json::Value = response.json().await?;
        let generated_text = response_json["generated_text"]
            .as_str()
            .ok_or_else(|| "Error: 'generated_text' field not found in LLM response".to_string())?;
        
        info!("Received LLM response (partial): {}", &generated_text[..std::cmp::min(generated_text.len(), 100)]);
        
        Ok(generated_text.to_string())
    }
}