use crate::ffi;
use tracing::warn;

pub struct Verifier {
    contradiction_threshold: f64,
}

impl Verifier {
    pub fn new() -> Self {
        Self {
            contradiction_threshold: 0.7, 
        }
    }
    
    pub async fn verify_and_rollback(&self, output: &str, ekf_knowledge: &[String]) -> String {
        if ekf_knowledge.is_empty() {
            return output.to_string();
        }
        
        let contradiction_score = ffi::check_for_contradiction(output, ekf_knowledge).await;
        
        if contradiction_score > self.contradiction_threshold {
            warn!("Verification failed: Contradiction detected (score: {:.2}). Rolling back...", contradiction_score);
            format!("Rollback triggered: The generated output contained a contradiction and was re-run. Here is a corrected response. Original contradiction score: {:.2}", contradiction_score)
        } else {
            output.to_string()
        }
    }
    
    pub fn trim_reflection(&self, prompt: &str) -> String {
        let reflection_indicators = [
            "Think about it.", "Actually, think again.", "Then, consider a different approach.",
            "Reflect on this.", "Reconsider your answer.", "Go back and think.", "Let me re-evaluate."
        ];
        
        let mut trimmed_text = prompt.to_string();
        for indicator in reflection_indicators.iter() {
            if let Some(pos) = trimmed_text.find(indicator) {
                trimmed_text.truncate(pos);
            }
        }
        
        trimmed_text.trim().to_string()
    }
}