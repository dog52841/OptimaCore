use nvml_rs::{enum_wrappers::device::ClockType, Nvml, NvmlDevice};
use once_cell::sync::OnceCell;
use std::error::Error;
use std::sync::{Arc, Mutex};
use tracing::info;

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
            Arc::new(Mutex::new(Nvml::init().expect(
                "Failed to initialize NVML. Ensure you have an NVIDIA GPU and drivers.",
            )))
        });

        let nvml = nvml_instance.lock().unwrap();
        let device = nvml.device_by_index(0)?;

        Ok(Self {
            utilization: 0.0,
            memory_bandwidth: 0.0,
            device,
        })
    }

    pub async fn get_utilization(&mut self) -> Result<f64, Box<dyn Error>> {
        let utilization = self.device.utilization_rates()?.gpu as f64;
        self.utilization = utilization;
        info!("Updated GPU utilization: {:.2}%", self.utilization);
        Ok(self.utilization)
    }

    pub async fn get_memory_bandwidth(&mut self) -> Result<f64, Box<dyn Error>> {
        // Compute theoretical maximum memory bandwidth based on current clock
        let mem_clock = self.device.clock_info(ClockType::Memory)? as f64; // MHz
        let bus_width = self.device.memory_bus_width()? as f64; // bits
        let theoretical_max_bandwidth = mem_clock * 2.0 * (bus_width / 8.0) / 1000.0; // GB/s

        let memory_util = self.device.utilization_rates()?.memory as f64;
        self.memory_bandwidth = memory_util / 100.0 * theoretical_max_bandwidth;
        info!(
            "Updated memory bandwidth: {:.2} GB/s",
            self.memory_bandwidth
        );
        Ok(self.memory_bandwidth)
    }
}
