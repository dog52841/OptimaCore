use nvml_wrapper::Nvml;
use tracing::info;
use std::error::Error;

/// GPU monitoring backed by NVIDIA's NVML. If NVML cannot be initialized
/// (e.g., running on a CPU-only machine), utilization and bandwidth values
/// gracefully fall back to zero.
pub struct GPUMonitor {
    nvml: Option<Nvml>,
    device_index: u32,
    utilization: f64,
    memory_bandwidth: f64,
}

impl GPUMonitor {
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        let nvml = match Nvml::init() {
            Ok(n) => {
                info!("NVML initialized for GPU monitoring");
                Some(n)
            }
            Err(e) => {
                info!("NVML unavailable: {}", e);
                None
            }
        };
        Ok(Self { nvml, device_index: 0, utilization: 0.0, memory_bandwidth: 0.0 })
    }

    pub async fn get_utilization(&mut self) -> f64 {
        if let Some(nvml) = &self.nvml {
            if let Ok(device) = nvml.device_by_index(self.device_index) {
                if let Ok(util) = device.utilization_rates() {
                    self.utilization = util.gpu as f64;
                    return self.utilization;
                }
            }
        }
        self.utilization = 0.0;
        self.utilization
    }

    pub async fn get_memory_bandwidth(&mut self) -> f64 {
        if let Some(nvml) = &self.nvml {
            if let Ok(device) = nvml.device_by_index(self.device_index) {
                if let Ok(util) = device.utilization_rates() {
                    let theoretical_max_bandwidth = 1000.0;
                    self.memory_bandwidth = util.memory as f64 / 100.0 * theoretical_max_bandwidth;
                    return self.memory_bandwidth;
                }
            }
        }
        self.memory_bandwidth = 0.0;
        self.memory_bandwidth
    }
}
