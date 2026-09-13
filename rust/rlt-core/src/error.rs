//! Error type for rlt-core.

use thiserror::Error;

/// Unified error type for the crate.
#[derive(Debug, Error)]
pub enum RltError {
    /// Errors from the candle tensor backend.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    /// Filesystem errors (checkpointing, corpora).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Invalid model or tokenizer configuration.
    #[error("config error: {0}")]
    Config(String),
    /// Tokenizer misuse (ids out of range, etc.).
    #[error("token error: {0}")]
    Token(String),
    /// Checkpoint save/load failures.
    #[error("checkpoint error: {0}")]
    Checkpoint(String),
    /// Recurrent-state misuse (stale position, cache mismatch).
    #[error("state error: {0}")]
    State(String),
    /// RL rollout/replay contract violations.
    #[error("replay error: {0}")]
    Replay(String),
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, RltError>;
