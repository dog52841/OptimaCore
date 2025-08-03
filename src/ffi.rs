use jlrs::prelude::*;
use once_cell::sync::OnceCell;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;
use std::error::Error;

static JULIA: OnceCell<Arc<Mutex<AsyncJulia>>> = OnceCell::new();
const JULIA_CODE: &str = include_str!("optimacore.jl");

pub fn init_julia() {
    JULIA.get_or_init(|| {
        info!("Initializing Julia runtime via jlrs...");
        let julia = unsafe {
            JuliaBuilder::new()
                .init_async::<jlrs::runtime::AsyncRuntime>()
                .unwrap()
                .async_run(|mut frame| async move {
                    frame.include_string(JULIA_CODE).unwrap();
                    Ok(())
                })
                .unwrap()
        };
        Arc::new(Mutex::new(julia))
    });
}

async fn run_julia_function_bool(name: &str, arg: &str) -> Result<bool, Box<dyn Error>> {
    let julia = JULIA.get().ok_or("Julia runtime not initialized")?.clone();
    let mut julia_locked = julia.lock().await;
    
    let result = julia_locked.async_run(|mut frame| async move {
        let func = Module::main(&frame).function(&frame, name)?;
        let julia_arg = Value::new(&mut frame, arg)?;
        let julia_result = func.call1(&mut frame, julia_arg)?;
        julia_result.cast::<bool>()
    }).await??;
    Ok(result)
}

async fn run_julia_function_contradiction(output: &str, facts: &[String]) -> Result<f64, Box<dyn Error>> {
    let julia = JULIA.get().ok_or("Julia runtime not initialized")?.clone();
    let mut julia_locked = julia.lock().await;
    
    let result = julia_locked.async_run(|mut frame| async move {
        let func = Module::main(&frame).function(&frame, "check_for_contradiction")?;
        let julia_output = Value::new(&mut frame, output)?;
        let julia_facts = Value::new(&mut frame, facts)?;
        let julia_result = func.call2(
            &mut frame, 
            julia_output, 
            julia_facts
        )?;
        julia_result.cast::<f64>()
    }).await??;
    Ok(result)
}

pub async fn detect_reflection_loop(prompt: &str) -> bool {
    run_julia_function_bool("detect_reflection_loop", prompt).await.unwrap_or(false)
}

pub async fn check_for_contradiction(output: &str, facts: &[String]) -> f64 {
    run_julia_function_contradiction(output, facts).await.unwrap_or(0.0)
}