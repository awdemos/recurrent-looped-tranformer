//! Checkpointing: safetensors weights (via candle's VarMap) + JSON config sidecar.
//!
//! A checkpoint is `PATH` (safetensors weights) plus `PATH` with extension
//! `st.json` (model configuration). Optimizer state is not persisted: resuming
//! training restarts the optimizer (warm restart), a documented v1 simplification.

use std::path::{Path, PathBuf};

use candle_core::Device;

use crate::{Result, Rlt, RltConfig, RltError};

fn config_path(weights: &Path) -> PathBuf {
    weights.with_extension("st.json")
}

/// Save model weights and configuration.
pub fn save_checkpoint(model: &Rlt, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }
    model.varmap.save(path)?;
    let json = serde_json::to_string_pretty(&model.config)
        .map_err(|e| RltError::Checkpoint(format!("config serialization: {e}")))?;
    std::fs::write(config_path(path), json)?;
    Ok(())
}

/// Load a checkpoint written by [`save_checkpoint`].
pub fn load_checkpoint(path: &Path, device: Device) -> Result<Rlt> {
    if !path.exists() {
        return Err(RltError::Checkpoint(format!(
            "no checkpoint at {}",
            path.display()
        )));
    }
    let cfg_path = config_path(path);
    let json = std::fs::read_to_string(&cfg_path)
        .map_err(|e| RltError::Checkpoint(format!("cannot read {}: {e}", cfg_path.display())))?;
    let config: RltConfig = serde_json::from_str(&json)
        .map_err(|e| RltError::Checkpoint(format!("config parse: {e}")))?;
    let mut model = Rlt::new(config, device)?;
    model.varmap.load(path)?;
    Ok(model)
}
