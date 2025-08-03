use tracing::info;
use nvml_rs::{Nvml, NvmlDevice};
use std::sync::{Arc, Mutex};
use once_cell::sync::OnceCell;
use std::error::Error;

static NVML: OnceCell<Arc<Mutex<Nvml>>> = OnceCell::new();

pub struct GPUMonitor {
    utilization: f64,
    memory_bandwidth: f64,
    device: NvmlDevice,
}

impl GPUMonitor {
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        let nvml_instance = NVML.get_or_init(|| {
            info!("Initializing NVML library for real GPU monitoring...");
            Arc::new(Mutex::new(Nvml::init().expect("Failed to initialize NVML. Ensure you have an NVIDIA GPU and drivers.")))
        });

        let nvml = nvml_instance.lock().unwrap();
        let device = nvml.device_by_index(0)?;
        
        Ok(Self {
            utilization: 0.0,
            memory_bandwidth: 0.0,
            device,
        })
    }
    
    pub async fn get_utilization(&mut self) -> f64 {
        let utilization = self.device.utilization_rates()?.gpu as f64;
        self.utilization = utilization;
        info!("Updated GPU utilization: {:.2}%", self.utilization);
        self.utilization
    }
    
    pub async fn get_memory_bandwidth(&mut self) -> f64 {
        let memory_info = self.device.memory_info()?;
        let memory_util = self.device.utilization_rates()?.memory as f64;
        let total_memory_gb = memory_info.total as f64 / (1024.0 * 1024.0 * 1024.0);
        
        let theoretical_max_bandwidth = 1000.0;
        let memory_bandwidth_gbps = memory_util / 100.0 * theoretical_max_bandwidth;

        self.memory_bandwidth = memory_bandwidth_gbps;
        info!("Updated memory bandwidth: {:.2} GB/s", self.memory_bandwidth);
        self.memory_bandwidth
    }
}