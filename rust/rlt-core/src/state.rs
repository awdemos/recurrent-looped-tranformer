//! Recurrent state: `H_t = (s_t, C_t^D)` plus encoder cache and encoder memory.

use candle_core::{Result, Tensor};

/// Key/value cache of one attention layer, layout (B, H, S, head_dim).
#[derive(Clone, Debug)]
pub struct LayerKv {
    /// Cached keys, post positional encoding.
    pub k: Tensor,
    /// Cached values.
    pub v: Tensor,
}

impl LayerKv {
    /// Number of cached positions.
    pub fn len(&self) -> usize {
        self.k.dim(2).unwrap_or(0)
    }

    /// True when no positions are cached.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append new keys/values along the sequence dim, returning a new cache.
    pub fn appended(&self, k: &Tensor, v: &Tensor) -> Result<Self> {
        Ok(Self {
            k: Tensor::cat(&[&self.k, k], 2)?,
            v: Tensor::cat(&[&self.v, v], 2)?,
        })
    }

    /// Drop the oldest entries so at most `max_len` positions remain.
    pub fn trimmed(&self, max_len: usize) -> Result<Self> {
        let len = self.len();
        if len <= max_len {
            return Ok(self.clone());
        }
        Ok(Self {
            k: self.k.narrow(2, len - max_len, max_len)?,
            v: self.v.narrow(2, len - max_len, max_len)?,
        })
    }
}

/// Encoder-derived global memory for one group, layout (1, H, S, head_dim).
/// Grows with the consumed prefix; decoder layers read only the prefix `M_{≤t}`.
#[derive(Clone, Debug)]
pub struct GroupKv {
    /// Memory keys (positional transformation already applied).
    pub k: Tensor,
    /// Memory values.
    pub v: Tensor,
}

impl GroupKv {
    /// Number of memory positions.
    pub fn len(&self) -> usize {
        self.k.dim(2).unwrap_or(0)
    }

    /// True when the group holds no memory positions.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append one position.
    pub fn appended(&self, k: &Tensor, v: &Tensor) -> Result<Self> {
        Ok(Self {
            k: Tensor::cat(&[&self.k, k], 2)?,
            v: Tensor::cat(&[&self.v, v], 2)?,
        })
    }
}

/// Complete recurrent state carried across tokens (paper §2.1, §2.3).
///
/// `H_t = (s_t, C_t^D)`; encoder continuation cache and encoder-derived memory are
/// maintained alongside. `H_0 = (s_star, ∅)` via [`crate::Rlt::initial_state`].
#[derive(Clone)]
pub struct RltState {
    /// Recurrent decoder output s_t, shape (1, d_model).
    pub s: Tensor,
    /// Per-decoder-layer sliding-window caches C_t^D (at most W−1 historical
    /// positions each; the current token's KV is transient during an update).
    pub decoder: Vec<Option<LayerKv>>,
    /// Per-encoder-layer causal KV cache C_t^E (full consumed prefix).
    pub encoder: Vec<Option<LayerKv>>,
    /// Encoder-derived global memory M, one entry per memory group.
    pub memory: Vec<GroupKv>,
    /// Next absolute position index (0-based); equals the number of consumed tokens.
    pub position: usize,
}
