//! Model configuration.

use serde::{Deserialize, Serialize};

use crate::{Result, RltError};

/// Configuration of an [`crate::Rlt`] model.
///
/// Defaults are demo-scale (`d_model=256`, 4+4 layers, window 64, one memory group,
/// tied weights). The paper's 48+48 configuration is reachable by overriding these
/// fields.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RltConfig {
    /// Residual width d.
    pub d_model: usize,
    /// Attention head count (must divide `d_model`).
    pub n_heads: usize,
    /// FFN hidden width.
    pub d_ff: usize,
    /// Encoder depth L_E.
    pub n_encoder_layers: usize,
    /// Decoder depth L_D.
    pub n_decoder_layers: usize,
    /// Sliding-window size W; includes the current token.
    pub window: usize,
    /// Encoder-memory group count G (paper §2.2).
    pub memory_groups: usize,
    /// Vocabulary size; byte-level tokenizer needs >= 258.
    pub vocab_size: usize,
    /// Tied configuration (paper §2.6): decoder SWA/FFN reuse encoder layer weights.
    pub tied: bool,
    /// Merge feedback scale α (paper Eq. 2.11).
    pub feedback_alpha: f64,
    /// RoPE table bound; positions beyond this are rejected.
    pub max_seq_len: usize,
}

impl Default for RltConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            n_heads: 8,
            d_ff: 1024,
            n_encoder_layers: 4,
            n_decoder_layers: 4,
            window: 64,
            memory_groups: 1,
            vocab_size: 258,
            tied: true,
            feedback_alpha: 0.1,
            max_seq_len: 4096,
        }
    }
}

impl RltConfig {
    /// Validate structural invariants.
    pub fn validate(&self) -> Result<()> {
        let err = |m: &str| RltError::Config(m.to_string());
        if self.n_heads == 0 {
            return Err(err("n_heads must be >= 1"));
        }
        if self.d_model == 0 || self.d_model % self.n_heads != 0 {
            return Err(err("n_heads must divide d_model"));
        }
        if self.n_encoder_layers == 0 || self.n_decoder_layers == 0 {
            return Err(err("layer counts must be >= 1"));
        }
        if self.n_decoder_layers > self.n_encoder_layers && self.tied {
            return Err(err("tied config requires n_decoder_layers <= n_encoder_layers"));
        }
        if self.window == 0 {
            return Err(err("window must be >= 1"));
        }
        if self.memory_groups == 0 {
            return Err(err("memory_groups must be >= 1"));
        }
        if self.d_ff == 0 {
            return Err(err("d_ff must be >= 1"));
        }
        if self.vocab_size < crate::tokenizer::MIN_VOCAB {
            return Err(err("vocab_size too small for the byte-level tokenizer"));
        }
        if self.max_seq_len == 0 {
            return Err(err("max_seq_len must be >= 1"));
        }
        if self.head_dim() % 2 != 0 {
            return Err(err("head_dim must be even (RoPE requirement)"));
        }
        if !self.feedback_alpha.is_finite() {
            return Err(err("feedback_alpha must be finite"));
        }
        Ok(())
    }

    /// Per-head width.
    ///
    /// Panics if `n_heads == 0`; call [`RltConfig::validate`] first.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Memory group read by decoder layer `layer` (paper: layer ℓ reads group g(ℓ)).
    ///
    /// Panics if `memory_groups == 0`; call [`RltConfig::validate`] first.
    pub fn memory_group_of(&self, layer: usize) -> usize {
        layer % self.memory_groups
    }
}
