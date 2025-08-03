use crate::hhtc::HHTCEngine;
use crate::ekf::EKFStorage;
use crate::verifier::Verifier;
use crate::gpu_monitor::GPUMonitor;
use crate::llm_integration::LLMClient;
use crate::ffi;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedResponse {
    pub output: String,
    pub compression_ratio: f64,
    pub reflection_trimmed: bool,
    pub ekf_knowledge: Vec<String>,
    pub bandwidth_saved: f64,
    pub gpu_utilization: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimaStats {
    pub total_requests: u64,
    pub avg_compression_ratio: f64,
    pub reflections_trimmed: u64,
    pub avg_gpu_utilization: f64,
    pub total_bandwidth_saved: f64,
}

pub struct OptimaCore {
    hhtc: Arc<Mutex<HHTCEngine>>,
    ekf: Arc<Mutex<EKFStorage>>,
    verifier: Arc<Mutex<Verifier>>,
    gpu_monitor: Arc<Mutex<GPUMonitor>>,
    llm_client: Arc<LLMClient>,
    
    request_count: u64,
    total_compression: f64,
    reflections_trimmed: u64,
    total_bandwidth_saved: f64,
    total_gpu_utilization: f64,
}

impl OptimaCore {
    pub async fn new(ekf_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let hhtc_engine = HHTCEngine::new(16, 1000).await?;
        let ekf_storage = EKFStorage::new(ekf_path).await?;
        let gpu_monitor = GPUMonitor::new().await?;
        let llm_client = LLMClient::new().await?;

        Ok(Self {
            hhtc: Arc::new(Mutex::new(hhtc_engine)),
            ekf: Arc::new(Mutex::new(ekf_storage)),
            verifier: Arc::new(Mutex::new(Verifier::new())),
            gpu_monitor: Arc::new(Mutex::new(gpu_monitor)),
            llm_client: Arc::new(llm_client),
            request_count: 0,
            total_compression: 0.0,
            reflections_trimmed: 0,
            total_bandwidth_saved: 0.0,
            total_gpu_utilization: 0.0,
        })
    }

    pub async fn process_request(&mut self, prompt: &str) -> Result<ProcessedResponse, Box<dyn std::error::Error>> {
        let reflection_detected = ffi::detect_reflection_loop(prompt).await;
        let trimmed_prompt = if reflection_detected {
            self.reflections_trimmed += 1;
            info!("Reflection loop detected. Trimming prompt...");
            let verifier = self.verifier.lock().await;
            verifier.trim_reflection(prompt)
        } else {
            prompt.to_string()
        };

        let mut monitor = self.gpu_monitor.lock().await;
        let gpu_utilization = monitor.get_utilization().await;
        let vram_bandwidth = monitor.get_memory_bandwidth().await;
        info!("GPU Utilization: {:.2}%, VRAM Bandwidth: {:.2} GB/s", gpu_utilization, vram_bandwidth);
        
        let (compressed_prompt, compression_ratio) = {
            let mut hhtc = self.hhtc.lock().await;
            hhtc.compress(&trimmed_prompt).await
        };
        info!("HHTC compression achieved: {:.2}% reduction", (1.0 - compression_ratio) * 100.0);
        
        let ekf_knowledge = {
            let mut ekf = self.ekf.lock().await;
            ekf.query(&compressed_prompt).await?
        };
        info!("EKF query returned {} knowledge snippets.", ekf_knowledge.len());
        
        let llm_output = {
            let llm = self.llm_client.clone();
            llm.generate(&compressed_prompt, &ekf_knowledge).await?
        };
        
        let final_output = {
            let mut verifier = self.verifier.lock().await;
            verifier.verify_and_rollback(&llm_output, &ekf_knowledge).await
        };
        
        let bandwidth_saved = vram_bandwidth * (1.0 - compression_ratio);
        
        self.request_count += 1;
        self.total_compression += 1.0 - compression_ratio;
        self.total_bandwidth_saved += bandwidth_saved;
        self.total_gpu_utilization += gpu_utilization;
        
        Ok(ProcessedResponse {
            output: final_output,
            compression_ratio,
            reflection_trimmed: reflection_detected,
            ekf_knowledge,
            bandwidth_saved,
            gpu_utilization,
        })
    }
    
    pub fn get_stats(&self) -> OptimaStats {
        let avg_compression_ratio = if self.request_count > 0 {
            self.total_compression / self.request_count as f64
        } else {
            0.0
        };
        
        let avg_gpu_utilization = if self.request_count > 0 {
            self.total_gpu_utilization / self.request_count as f64
        } else {
            0.0
        };
        
        OptimaStats {
            total_requests: self.request_count,
            avg_compression_ratio,
            reflections_trimmed: self.reflections_trimmed,
            avg_gpu_utilization,
            total_bandwidth_saved: self.total_bandwidth_saved,
        }
    }
}