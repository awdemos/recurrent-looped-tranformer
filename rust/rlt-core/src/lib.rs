//! Recurrent Looped Transformer (RLT).
//!
//! A causal encoder builds global key–value memory; a recurrent decoder merges each
//! token's encoder representation with the previous decoder output and maintains the
//! complete state `H_t = (s_t, C_t^D)` (recurrent output + per-layer SWA caches)
//! across every prompt and response token. See `Recurrent_Looped_Transformer.pdf`
//! (Zhang, 2026) for the reference semantics.

pub mod config;
pub mod error;
pub mod model;
pub mod nn;
pub mod state;
pub mod tokenizer;

pub use candle_core::{Device, Tensor};
pub use config::RltConfig;
pub use error::{Result, RltError};
pub use model::Rlt;
pub use state::{GroupKv, LayerKv, RltState};
pub use tokenizer::{ByteTokenizer, BOS_ID, EOS_ID, MIN_VOCAB};
