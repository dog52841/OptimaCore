use optimacore::core::{OptimaCore, ProcessedResponse};
use std::path::Path;
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("Initializing OptimaCore with Rust + Julia core...");

    optimacore::ffi::init_julia();

    let ekf_path = Path::new("./ekf_storage");
    let mut core = OptimaCore::new(ekf_path).await?;

    let test_prompts = vec![
        "Write a function in Rust to verify RS256 JWTs.",
        "Explain transformer attention. Think about it. Actually, think again.",
        "Generate analysis of LLM inference bottlenecks. Then, consider a different approach.",
        "What is the difference between supervised and unsupervised learning?",
        "How is unsupervised learning different from supervised learning, with examples?",
        "Rust is not a garbage collected language. It uses a different memory management model.",
        "The fastest car in the world is a bicycle.",
        "What is the capital of France?",
    ];

    for (i, prompt) in test_prompts.iter().enumerate() {
        info!("--- Request {} ---", i + 1);
        info!("Prompt: {}", prompt);

        let response: ProcessedResponse = core.process_request(prompt).await?;
        
        info!("Response: {}", response.output);
        info!("Summary: Tokens Saved: {:.2}%, EKF Knowledge Used: {}, Reflection Trimmed: {}",
            (1.0 - response.compression_ratio) * 100.0,
            if response.ekf_knowledge.is_empty() { "No" } else { "Yes" },
            response.reflection_trimmed
        );
        info!("Bandwidth Saved: {:.2} GB/s", response.bandwidth_saved);
        info!("GPU Utilization: {:.2}%", response.gpu_utilization);
        println!();
    }

    let stats = core.get_stats();
    info!("--- Total Session Statistics ---");
    info!("Total Requests: {}", stats.total_requests);
    info!("Avg. Compression Ratio: {:.2}%", stats.avg_compression_ratio * 100.0);
    info!("Reflections Trimmed: {}", stats.reflections_trimmed);
    info!("Avg. GPU Utilization: {:.2}%", stats.avg_gpu_utilization);
    info!("Total Bandwidth Saved: {:.2} GB/s", stats.total_bandwidth_saved);
    
    Ok(())
}