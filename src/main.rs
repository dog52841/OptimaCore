use optimacore::core::{OptimaCore, ProcessedResponse};
use std::env;
use std::path::Path;
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    optimacore::ffi::init_julia();

    let prompt = env::args().skip(1).collect::<Vec<String>>().join(" ");
    if prompt.is_empty() {
        eprintln!("Usage: optimacore <prompt>");
        return Ok(());
    }

    let ekf_path = Path::new("./ekf_storage");
    let mut core = OptimaCore::new(ekf_path).await?;

    let response: ProcessedResponse = core.process_request(&prompt).await?;
    println!("{}", response.output);
    info!("Tokens Saved: {:.2}%", (1.0 - response.compression_ratio) * 100.0);
    info!("EKF Knowledge Used: {}", if response.ekf_knowledge.is_empty() { "No" } else { "Yes" });
    info!("Reflection Trimmed: {}", response.reflection_trimmed);
    info!("Bandwidth Saved: {:.2} GB/s", response.bandwidth_saved);
    info!("GPU Utilization: {:.2}%", response.gpu_utilization);

    Ok(())
}

