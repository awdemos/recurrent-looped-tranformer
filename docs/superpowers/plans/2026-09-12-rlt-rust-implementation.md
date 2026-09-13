# RLT Rust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Recurrent Looped Transformer (RLT) from `Recurrent_Looped_Transformer.pdf` as the published library crate `rlt-core` (candle-based, full BPTT + RL replay) plus the binary crate `rlt-cli` (train/generate), verify per spec §8, publish both to crates.io.

**Architecture:** Cargo workspace in `rust/` with two crates. `rlt-core` implements the paper: parallel causal encoder → global KV memory groups → gated merge → recurrent decoder (SWA → cross-attn → FFN) with complete state `H_t = (s_t, C_t^D)`, plus incremental `step`, full-BPTT `train_step`, and `replay`. `rlt-cli` is a pure consumer of the public API. The decoder is ALWAYS sequential (prefill replays every token); only the encoder batches.

**Tech Stack:** Rust 1.98, edition 2021, candle-core 0.11 + candle-nn 0.11 (autograd, CPU default / optional CUDA), serde/serde_json, thiserror 2, rand 0.10, clap 4, anyhow 1.

**Spec:** `docs/superpowers/specs/2026-09-12-rlt-rust-implementation-design.md` (authoritative for behavior).

**Conventions used throughout:**
- All tensors f32 on `Device::Cpu` in tests.
- `Rlt::new(config, device)` creates its own `VarMap` internally; the store is exposed
  as the public field `varmap`. `VarMap`/`Device` do NOT implement `Default` — never
  write `Default::default()` for them; use `Device::Cpu` and `model.varmap`.
- Norms are unweighted RMSNorm (`NORM_EPS = 1e-5`); all Linear layers bias-free except merge gate bias `b_g` (paper explicit).
- Positions are 0-based internally; RoPE offset = position index.
- Paper eq. references: memory Eq. 2.2, merge Eq. 2.9–2.11, decoder block Eq. 2.12–2.16, readout Eq. 2.6.
- Candle 0.11 API notes (verified by the Task 2 probe in `rust/rlt-core/tests/api_probe.rs`):
  `softmax`/`log_softmax`/`sigmoid` are free functions in `candle_nn::ops`;
  `Init::Uniform { lo, up }` (field is `up`, not `hi`); `VarBuilder::get`/`get_with_hints`
  return `Tensor` directly (NO `.as_tensor()`); `AdamW::new(vars, ParamsAdamW { lr, ..Default::default() })`;
  mutate a parameter via `model.varmap.data().lock().unwrap().get(name)` (a `Var`) + `.set(&tensor)`;
  4-D tensor extraction has no `to_vec4` — use the `t4` reshape+`to_vec3` helper in tests.
- The module tree is wired incrementally: `lib.rs` gains `pub mod` lines and
  re-exports task by task so the workspace builds green at every commit. There is no
  `attention.rs` module — attention primitives live in `nn.rs`.

---

### Task 1: Scaffold the workspace

**Files:**
- Create: `rust/Cargo.toml`, `rust/rlt-core/Cargo.toml`, `rust/rlt-core/src/lib.rs`,
  `rust/rlt-cli/Cargo.toml`, `rust/rlt-cli/src/main.rs`, `rust/.gitignore`
- Copy: MIT + Apache-2.0 license texts into `rust/rlt-core/LICENSE-MIT`,
  `rust/rlt-core/LICENSE-APACHE`, `rust/rlt-cli/LICENSE-MIT`, `rust/rlt-cli/LICENSE-APACHE`

- [ ] **Step 1: Write `rust/Cargo.toml`**

```toml
[workspace]
resolver = "2"
members = ["rlt-core", "rlt-cli"]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
authors = ["Yifan Zhang"]
homepage = "https://github.com/awdemos/recurrent-looped-tranformer"
repository = "https://github.com/awdemos/recurrent-looped-tranformer"

[workspace.dependencies]
candle-core = "0.11"
candle-nn = "0.11"
```

- [ ] **Step 2: Write `rust/rlt-core/Cargo.toml`**

```toml
[package]
name = "rlt-core"
description = "Recurrent Looped Transformer (RLT): causal encoder with global KV memory plus a recurrent decoder with sliding-window attention. Full-BPTT training and RL replay. Pure Rust (candle)."
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true
homepage.workspace = true
repository.workspace = true
documentation = "https://docs.rs/rlt-core"
readme = "README.md"
categories = ["science", "algorithms"]
keywords = ["transformer", "recurrent", "attention", "rlt", "language-model"]

[dependencies]
candle-core.workspace = true
candle-nn.workspace = true
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
rand = "0.10"

[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda"]
```

- [ ] **Step 3: Write `rust/rlt-cli/Cargo.toml`**

```toml
[package]
name = "rlt-cli"
description = "CLI for the Recurrent Looped Transformer (rlt-core): initialize checkpoints, train on a corpus with full BPTT, generate text."
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true
homepage.workspace = true
repository.workspace = true
documentation = "https://docs.rs/rlt-cli"
readme = "README.md"
categories = ["science", "command-line-utilities"]
keywords = ["rlt", "transformer", "language-model", "cli"]

[[bin]]
name = "rlt"
path = "src/main.rs"

[dependencies]
rlt-core = { version = "0.1.0", path = "../rlt-core" }
candle-nn.workspace = true
clap = { version = "4", features = ["derive"] }
anyhow = "1"
rand = "0.10"
```

- [ ] **Step 4: Write `rust/rlt-core/src/lib.rs`** (docs only — modules are wired in
later tasks so every commit builds)

```rust
//! Recurrent Looped Transformer (RLT).
//!
//! A causal encoder builds global key–value memory; a recurrent decoder merges each
//! token's encoder representation with the previous decoder output and maintains the
//! complete state `H_t = (s_t, C_t^D)` (recurrent output + per-layer SWA caches)
//! across every prompt and response token. See `Recurrent_Looped_Transformer.pdf`
//! (Zhang, 2026) for the reference semantics.
```

- [ ] **Step 5: Write `rust/rlt-cli/src/main.rs`**

```rust
fn main() {
    println!("rlt: not yet implemented");
}
```

- [ ] **Step 6: Write `rust/.gitignore`**

```
/target
```

- [ ] **Step 7: Copy license texts.** Fetch the standard MIT and Apache-2.0 texts
(e.g. from <https://opensource.org/licenses/MIT> and
<https://www.apache.org/licenses/LICENSE-2.0.txt>) into all four `LICENSE-*` files
listed above. In the MIT license, set the copyright line to
`Copyright (c) 2026 Yifan Zhang`.

- [ ] **Step 8: Build the empty scaffold**

Run: `cd rust && cargo build 2>&1 | tail -5`
Expected: `Finished` with exit code 0.

- [ ] **Step 9: Commit**

```bash
git add rust/
git commit -m "rust: scaffold workspace (rlt-core lib + rlt-cli bin)"
```

---

### Task 2: Candle 0.11 API probe

Every candle call the implementation relies on, verified in one integration test.
If candle 0.11 drifted from these signatures, fix the probe first and adjust later
tasks accordingly.

**Files:**
- Test: `rust/rlt-core/tests/api_probe.rs`

- [ ] **Step 1: Write the probe test**

```rust
//! Compile+run probe for every candle API this crate depends on.
use candle_core::{D, Device, DType, Tensor, Var};
use candle_nn::{AdamW, Embedding, Linear, Module, Optimizer, VarBuilder, VarMap, Init};

fn dev() -> Device { Device::Cpu }

#[test]
fn tensor_ops_probe() -> candle_core::Result<()> {
    let d = dev();
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &d)?;
    let _ = x.powf(2.)?.mean(D::Minus1)?.sqrt()?;
    let _ = x.broadcast_add(&Tensor::new(1e-5f32, &d)?)?;
    let _ = x.affine(2.0, 1.0)?.sigmoid()?;
    let _ = x.matmul(&x)?;
    let _ = x.softmax(D::Minus1)?.log_softmax(D::Minus1)?;
    let idx = Tensor::from_vec(vec![1u32, 0], (2, 1), &d)?;
    let _ = x.gather(&idx, 1)?;
    let _ = x.narrow(0, 0, 1)?;
    let _ = Tensor::cat(&[&x, &x], 0)?;
    let _ = Tensor::stack(&[&x, &x], 0)?;
    let _ = Tensor::ones((2, 2), DType::F32, &d)?;
    let _ = Tensor::arange(0u32, 4, &d)?.to_dtype(DType::F32)?.cos()?.sin()?;
    let _ = x.to_vec2::<f32>()?;
    let _ = x.unsqueeze(0)?;
    let _ = x.transpose(0, 1)?;
    let _ = x.reshape((4,))?;
    let _ = x.sum_all()?;
    let _ = x.broadcast_div(&Tensor::new(2.0f32, &d)?)?;
    let _ = x.dim(D::Minus1)?;
    let _ = x.dims2()?;
    assert_eq!(x.to_scalar::<f32>()?, 1.0);
    let _ = Tensor::new(vec![vec![0f32, f32::NEG_INFINITY], vec![0.0, 0.0]], &d)?;
    let _ = Tensor::rand(-1f32, 1f32, (2, 3), &d)?;
    Ok(())
}

#[test]
fn var_and_optimizer_probe() -> candle_core::Result<()> {
    let d = dev();
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &d);
    let w = vb.get_with_hints((2, 2), "w", Init::Uniform { lo: -0.1, hi: 0.1 })?;
    let lin = Linear::new(w.as_tensor().clone(), None);
    let emb = Embedding::new(vb.get((4, 2), "emb")?.as_tensor().clone(), 2);
    let b = vb.pp("merge").get((2,), "b")?;
    assert!(b.as_tensor().dims1()? == 2);
    // VarMap::data() must be public for gradient-check tests
    let _names: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
    // Var::set must exist (finite-difference tests mutate parameters)
    b.set(&Tensor::new(&[0.5f32, 0.25], &d)?)?;
    let ids = Tensor::from_vec(vec![0u32, 1, 2], (3,), &d)?;
    let _ = emb.forward(&ids)?;
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &d);
    let y = lin.forward(&x)?;
    let mut opt = AdamW::new(0.01, vm.all_vars())?;
    let loss = y.sqr()?.sum_all()?;
    opt.backward_step(&loss)?;
    let grads = loss.backward()?;
    let _g = grads.get(w.as_tensor());
    Ok(())
}
```

- [ ] **Step 2: Run the probe**

Run: `cd rust && cargo test -p rlt-core --test api_probe 2>&1 | tail -15`
Expected: `test result: ok. 2 passed` — exit code 0.
If any line fails to compile: adjust the call site to the candle 0.11 API (check
`~/.cargo/registry/src/` for the candle-core source), and apply the same fix
everywhere the plan uses that call.

- [ ] **Step 3: Commit**

```bash
git add rust/rlt-core/tests/api_probe.rs
git commit -m "rust: candle 0.11 API probe"
```

---

### Task 3: `error.rs` and `config.rs`

**Files:**
- Create: `rust/rlt-core/src/error.rs`, `rust/rlt-core/src/config.rs`
- Modify: `rust/rlt-core/src/lib.rs` — add `pub mod config; pub mod error;` and
  `pub use config::RltConfig; pub use error::{Result, RltError};`
- Test: `rust/rlt-core/tests/config.rs`

- [ ] **Step 1: Write the failing config test**

```rust
use rlt_core::RltConfig;

#[test]
fn defaults_validate() {
    let cfg = RltConfig::default();
    cfg.validate().unwrap();
    assert_eq!(cfg.head_dim(), cfg.d_model / cfg.n_heads);
}

#[test]
fn rejects_invalid() {
    let mut cfg = RltConfig::default();
    cfg.n_heads = 3; // 256 % 3 != 0
    assert!(cfg.validate().is_err());
    let mut cfg = RltConfig::default();
    cfg.window = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn memory_group_mapping() {
    let cfg = RltConfig { memory_groups: 3, n_decoder_layers: 7, ..Default::default() };
    assert_eq!(cfg.memory_group_of(0), 0);
    assert_eq!(cfg.memory_group_of(5), 2);
    assert_eq!(cfg.memory_group_of(6), 0);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd rust && cargo test -p rlt-core --test config 2>&1 | tail -5`
Expected: compile error: `RltConfig` not found / no `validate`.

- [ ] **Step 3: Implement `rust/rlt-core/src/error.rs`**

```rust
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
```

- [ ] **Step 4: Implement `rust/rlt-core/src/config.rs`**

```rust
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
        Ok(())
    }

    /// Per-head width.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Memory group read by decoder layer `layer` (paper: layer ℓ reads group g(ℓ)).
    pub fn memory_group_of(&self, layer: usize) -> usize {
        layer % self.memory_groups
    }
}
```

Note: `config.rs` references `crate::tokenizer::MIN_VOCAB` — create
`rust/rlt-core/src/tokenizer.rs` with ONLY the constant now so the build stays
green (full tokenizer arrives in Task 4):

```rust
//! Byte-level tokenizer with BOS/EOS specials. (Full implementation in Task 4.)
pub const MIN_VOCAB: usize = 258;
```

and add `pub mod tokenizer;` to lib.rs in this task.

- [ ] **Step 5: Run tests**

Run: `cd rust && cargo test -p rlt-core --test config 2>&1 | tail -5`
Expected: `test result: ok. 3 passed`.

- [ ] **Step 6: Commit**

```bash
git add rust/rlt-core/src/error.rs rust/rlt-core/src/config.rs rust/rlt-core/src/tokenizer.rs rust/rlt-core/src/lib.rs rust/rlt-core/tests/config.rs
git commit -m "rust(rlt-core): error type + model config"
```

---

### Task 4: `tokenizer.rs`

**Files:**
- Modify: `rust/rlt-core/src/tokenizer.rs`
- Modify: `rust/rlt-core/src/lib.rs` — add `pub use tokenizer::{ByteTokenizer, BOS_ID, EOS_ID, MIN_VOCAB};`
- Test: `rust/rlt-core/tests/tokenizer.rs`

- [ ] **Step 1: Write the failing test**

```rust
use rlt_core::{ByteTokenizer, BOS_ID, EOS_ID};

#[test]
fn roundtrip_with_specials() {
    let tok = ByteTokenizer::new(258).unwrap();
    let text = "Hello, 世界 → RLT\n";
    let ids = tok.encode(text, true, true);
    assert_eq!(ids[0], BOS_ID);
    assert_eq!(*ids.last().unwrap(), EOS_ID);
    assert_eq!(tok.decode(&ids), text);
}

#[test]
fn decode_skips_specials() {
    let tok = ByteTokenizer::new(258).unwrap();
    assert_eq!(tok.decode(&[BOS_ID, b'a' as u32, b'b' as u32, EOS_ID]), "ab");
}

#[test]
fn rejects_tiny_vocab() {
    assert!(ByteTokenizer::new(100).is_err());
}
```

- [ ] **Step 2: Run to verify failure** — `cd rust && cargo test -p rlt-core --test tokenizer 2>&1 | tail -3`; expect compile error (`ByteTokenizer` not found).

- [ ] **Step 3: Implement `rust/rlt-core/src/tokenizer.rs` (full version)**

```rust
//! Byte-level tokenizer with BOS/EOS specials. Keeps demos self-contained.

use crate::{Result, RltError};

/// Beginning-of-sequence token id (byte-level: 256).
pub const BOS_ID: u32 = 256;
/// End-of-sequence token id (byte-level: 257).
pub const EOS_ID: u32 = 257;
/// Minimum vocab size for this tokenizer (all bytes + BOS + EOS).
pub const MIN_VOCAB: usize = 258;

/// Byte-level tokenizer: token id = byte value; specials live above 255.
#[derive(Clone, Debug)]
pub struct ByteTokenizer {
    vocab_size: usize,
}

impl ByteTokenizer {
    /// Create a tokenizer; `vocab_size` must be >= [`MIN_VOCAB`].
    pub fn new(vocab_size: usize) -> Result<Self> {
        if vocab_size < MIN_VOCAB {
            return Err(RltError::Token(format!(
                "vocab_size {vocab_size} < {MIN_VOCAB} (need all byte values + BOS + EOS)"
            )));
        }
        Ok(Self { vocab_size })
    }

    /// Encode text to token ids, optionally adding BOS/EOS.
    pub fn encode(&self, text: &str, with_bos: bool, with_eos: bool) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() + 2);
        if with_bos {
            ids.push(BOS_ID);
        }
        ids.extend(text.as_bytes().iter().map(|b| u32::from(*b)));
        if with_eos {
            ids.push(EOS_ID);
        }
        ids
    }

    /// Decode token ids back to text; special tokens (>= 256) are skipped and
    /// invalid ids are replaced with the replacement character.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter_map(|&t| u8::try_from(t).ok())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Vocabulary size this tokenizer was built for.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
```

- [ ] **Step 4: Run tests** — expect `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add rust/rlt-core/src/tokenizer.rs rust/rlt-core/src/lib.rs rust/rlt-core/tests/tokenizer.rs
git commit -m "rust(rlt-core): byte-level tokenizer"
```

---

### Task 5: `state.rs` — cache and state types

**Files:**
- Create: `rust/rlt-core/src/state.rs`
- Modify: `rust/rlt-core/src/lib.rs` — add `pub mod state;` and
  `pub use state::{GroupKv, LayerKv, RltState};`

- [ ] **Step 1: Implement `rust/rlt-core/src/state.rs`**

```rust
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
```

- [ ] **Step 2: Build** — `cd rust && cargo build -p rlt-core 2>&1 | tail -3`; expect success.

- [ ] **Step 3: Commit**

```bash
git add rust/rlt-core/src/state.rs rust/rlt-core/src/lib.rs
git commit -m "rust(rlt-core): recurrent state + KV cache types"
```

---

### Task 6: `nn.rs` — tensor primitives

**Files:**
- Create: `rust/rlt-core/src/nn.rs`
- Modify: `rust/rlt-core/src/lib.rs` — add `pub mod nn;`
- Test: `rust/rlt-core/tests/nn.rs`

- [ ] **Step 1: Write the failing tests**

```rust
use candle_core::{D, Device, Tensor};
use rlt_core::nn;
use rlt_core::nn::RotaryEmbedding;

fn dev() -> Device { Device::Cpu }

/// 4-D extraction (candle has no `to_vec4`): reshape to 3-D and regroup.
fn t4(t: &Tensor) -> Vec<Vec<Vec<Vec<f32>>>> {
    let (a, b, c, d) = t.dims4().unwrap();
    t.reshape((a * b, c, d)).unwrap()
        .to_vec3::<f32>().unwrap()
        .chunks(b)
        .map(|c| c.to_vec())
        .collect()
}

#[test]
fn rms_norm_matches_manual() {
    let x = Tensor::new(&[[3.0f32, 4.0], &[1.0, 2.0]], &dev()).unwrap();
    let y = nn::rms_norm(&x, 1e-5).unwrap().to_vec2::<f32>().unwrap();
    let ms0 = (9.0 + 16.0) / 2.0;
    for (got, want) in y[0].iter().zip([3.0 / ms0.sqrt(), 4.0 / ms0.sqrt()]) {
        assert!((got - want).abs() < 1e-5);
    }
}

#[test]
fn rope_known_angles() {
    // dim=4, base=10000: pair angles at pos 1 are [1.0, 0.01]
    let rotary = RotaryEmbedding::new(4, 8, &dev()).unwrap();
    let x = Tensor::new(&[[[[1.0f32, 0.0, 1.0, 0.0]]]], &dev()).unwrap();
    let (q, _) = rotary.apply_qk(&x, &x, 1).unwrap();
    let v = t4(&q);
    let (c, s) = (1.0f32.cos(), 1.0f32.sin());
    assert!((v[0][0][0][0] - (c - s)).abs() < 1e-5, "got {}", v[0][0][0][0]);
    assert!(v[0][0][0][1].abs() < 1e-6);
    assert!((v[0][0][0][2] - (c + s)).abs() < 1e-5);
    assert!(v[0][0][0][3].abs() < 1e-6);
}

#[test]
fn rope_preserves_norm() {
    let rotary = RotaryEmbedding::new(8, 16, &dev()).unwrap();
    let x = Tensor::rand(-1.0f32, 1.0, (1, 2, 5, 8), &dev()).unwrap();
    let (q, _) = rotary.apply_qk(&x, &x, 3).unwrap();
    let a = x.powf(2.)?.sum(D::Minus1)?.sqrt()?;
    let b = q.powf(2.)?.sum(D::Minus1)?.sqrt()?;
    let diff = (a - b)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-4);
}

#[test]
fn sdpa_matches_naive() {
    let d = dev();
    let q = Tensor::rand(-1.0f32, 1.0, (1, 2, 3, 4), &d).unwrap();
    let k = Tensor::rand(-1.0f32, 1.0, (1, 2, 5, 4), &d).unwrap();
    let v = Tensor::rand(-1.0f32, 1.0, (1, 2, 5, 4), &d).unwrap();
    let got = t4(&nn::sdpa(&q, &k, &v, None).unwrap());
    let qv = t4(&q);
    let kv = t4(&k);
    let vv = t4(&v);
    let scale = 1.0 / 4.0f32.sqrt();
    for h in 0..2 {
        for i in 0..3 {
            let mut scores = vec![0f32; 5];
            for j in 0..5 {
                let mut dot = 0f32;
                for x in 0..4 {
                    dot += qv[0][h][i][x] * kv[0][h][j][x];
                }
                scores[j] = dot * scale;
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            for x in 0..4 {
                let mut want = 0f32;
                for j in 0..5 {
                    want += (exps[j] / z) * vv[0][h][j][x];
                }
                assert!((got[0][h][i][x] - want).abs() < 1e-4);
            }
        }
    }
}

#[test]
fn causal_mask_blocks_future() {
    let m = nn::causal_mask(3, &dev()).unwrap();
    let v = t4(&m);
    assert_eq!(v[0][0][0][1], f32::NEG_INFINITY);
    assert_eq!(v[0][0][0][2], f32::NEG_INFINITY);
    assert_eq!(v[0][0][1][2], f32::NEG_INFINITY);
    assert_eq!(v[0][0][2][0], 0.0);
    assert_eq!(v[0][0][0][0], 0.0);
}
```

- [ ] **Step 2: Run to verify failure** — `cd rust && cargo test -p rlt-core --test nn 2>&1 | tail -3`; expect compile error (`rlt_core::nn` not found).

- [ ] **Step 3: Implement `rust/rlt-core/src/nn.rs`**

```rust
//! Tensor-level primitives: RMSNorm, rotary embeddings, attention.

use candle_core::{bail, D, Device, Result, Tensor};

/// RMSNorm epsilon used throughout the model.
pub const NORM_EPS: f32 = 1e-5;

/// Unweighted RMSNorm over the last dimension.
pub fn rms_norm(x: &Tensor, eps: f32) -> Result<Tensor> {
    let ms = x.powf(2.)?.mean(D::Minus1)?;
    let denom = ms.broadcast_add(&Tensor::new(eps, x.device())?)?.sqrt()?;
    x.broadcast_div(&denom)
}

/// Rotates pairs of channels: `[-x_{d/2..}, x_{0..d/2}]` (standard RoPE pairing).
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?.affine(-1.0, 0.0)?;
    Tensor::cat(&[&x2, &x1], D::Minus1)
}

/// Cached rotary embedding table (RoPE, base 10000).
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    /// Build tables for `dim` channels (must be positive even) up to `max_seq_len`.
    pub fn new(dim: usize, max_seq_len: usize, device: &Device) -> Result<Self> {
        if dim == 0 || dim % 2 != 0 {
            bail!("rotary dim must be a positive even number, got {dim}");
        }
        let base = 10_000f32;
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| base.powf(-(i as f32) * 2.0 / dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (dim / 2,), device)?;
        let pos = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(candle_core::DType::F32)?;
        let freqs = pos.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?; // (S, dim/2)
        Ok(Self { cos: freqs.cos()?, sin: freqs.sin()? })
    }

    /// Apply rotation to a single tensor (B, H, S, D) at absolute `offset`.
    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let seq = x.dim(2)?;
        let cos = self.cos.narrow(0, offset, seq)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, offset, seq)?.unsqueeze(0)?.unsqueeze(0)?;
        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?; // (1,1,S,D)
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?;
        x.broadcast_mul(&cos)?
            .broadcast_add(&rotate_half(x)?.broadcast_mul(&sin)?)
    }

    /// Apply rotation to query and key (B, H, S, D) at absolute `offset`.
    pub fn apply_qk(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        Ok((self.apply(q, offset)?, self.apply(k, offset)?))
    }
}

/// Scaled dot-product attention.
///
/// `q`: (B, H, Sq, D); `k`, `v`: (B, H, Sk, D). `mask` is an additive broadcastable
/// tensor (e.g. (1,1,Sq,Sk) from [`causal_mask`]), or `None` when every key is valid.
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let d = q.dim(D::Minus1)? as f64;
    let mut scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
    scores = scores.affine(1.0 / d.sqrt(), 0.0)?;
    if let Some(m) = mask {
        scores = scores.broadcast_add(m)?;
    }
    candle_nn::ops::softmax(&scores, D::Minus1)?.matmul(v)
}

/// Additive causal mask (1, 1, T, T): 0 on/below the diagonal, −∞ above.
pub fn causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mut rows = Vec::with_capacity(t);
    for i in 0..t {
        let mut row = vec![0f32; t];
        for j in (i + 1)..t {
            row[j] = f32::NEG_INFINITY;
        }
        rows.push(row);
    }
    Tensor::new(rows, device)?.unsqueeze(0)?.unsqueeze(0)
}

/// (B, S, D) -> (B, H, S, D/h).
pub fn split_heads(x: &Tensor, n_heads: usize) -> Result<Tensor> {
    let (b, s, d) = x.dims3()?;
    x.reshape((b, s, n_heads, d / n_heads))?.transpose(1, 2)
}

/// (B, H, S, D/h) -> (B, S, D).
pub fn merge_heads(x: &Tensor) -> Result<Tensor> {
    let (b, h, s, hd) = x.dims4()?;
    x.transpose(1, 2)?.reshape((b, s, h * hd))
}
```

- [ ] **Step 4: Run tests** — expect `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add rust/rlt-core/src/nn.rs rust/rlt-core/src/lib.rs rust/rlt-core/tests/nn.rs
git commit -m "rust(rlt-core): rmsnorm, rope, attention primitives"
```

---

### Task 7: `model.rs` — encoder + `Rlt` core

**Files:**
- Create: `rust/rlt-core/src/model.rs`
- Modify: `rust/rlt-core/src/lib.rs` — add `pub mod model;` and `pub use model::Rlt;`
  plus `pub use candle_core::{Device, Tensor};`
- Test: `rust/rlt-core/tests/encoder.rs`

- [ ] **Step 1: Write the failing encoder test**

```rust
use candle_core::Device;
use rlt_core::{RltConfig, Rlt};

fn tiny_config() -> RltConfig {
    RltConfig {
        d_model: 32,
        n_heads: 2,
        d_ff: 64,
        n_encoder_layers: 2,
        n_decoder_layers: 2,
        window: 4,
        memory_groups: 1,
        vocab_size: 258,
        tied: false,
        feedback_alpha: 0.1,
        max_seq_len: 64,
    }
}

#[test]
fn encode_parallel_matches_incremental() {
    let cfg = tiny_config();
    let model = Rlt::new(cfg, Device::Cpu).unwrap();
    let tokens: Vec<u32> = vec![5, 9, 1, 2, 3, 8, 7, 6, 4, 0];
    let (e, enc_cache, _mem) = model.encode(&tokens).unwrap();
    assert_eq!(e.dims2().unwrap(), (10, 32));
    let mut state_enc: Vec<Option<rlt_core::LayerKv>> = vec![None; cfg.n_encoder_layers];
    let mut e_steps = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        let e_t = model.encode_step(t, pos, &mut state_enc).unwrap();
        e_steps.push(e_t.squeeze(0).unwrap());
    }
    let e_inc = candle_core::Tensor::stack(&e_steps, 0).unwrap();
    let diff = (e - e_inc).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(diff < 1e-4, "parallel vs incremental encoder mismatch: {diff}");
    for c in enc_cache.iter().flatten() {
        assert_eq!(c.len(), tokens.len());
    }
}
```

- [ ] **Step 2: Run to verify failure** — compile error: `Rlt` has no `encode`.

- [ ] **Step 3: Implement `rust/rlt-core/src/model.rs`**

```rust
//! The RLT model: encoder stack, decoder stack, merge, readout.

use candle_core::{D, Device, DType, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, VarMap, Init};

use crate::config::RltConfig;
use crate::nn::{self, RotaryEmbedding, NORM_EPS};
use crate::state::{GroupKv, LayerKv};
use crate::RltError;

/// Bias-free linear with uniform init.
fn linear(vb: VarBuilder, in_dim: usize, out_dim: usize) -> Result<Linear> {
    let weight = vb.get_with_hints(
        (in_dim, out_dim),
        "weight",
        Init::Uniform { lo: -0.02, up: 0.02 },
    )?;
    Ok(Linear::new(weight, None))
}

/// One encoder layer: causal MHA (RoPE) + FFN, pre-norm with residuals.
#[derive(Clone)]
pub struct EncoderBlock {
    /// Query projection.
    pub q_proj: Linear,
    /// Key projection.
    pub k_proj: Linear,
    /// Value projection.
    pub v_proj: Linear,
    /// Attention output projection.
    pub o_proj: Linear,
    /// FFN up projection.
    pub ff1: Linear,
    /// FFN down projection.
    pub ff2: Linear,
    /// Head count.
    pub n_heads: usize,
}

impl EncoderBlock {
    fn new(cfg: &RltConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        Ok(Self {
            q_proj: linear(vb.pp("q_proj"), d, d)?,
            k_proj: linear(vb.pp("k_proj"), d, d)?,
            v_proj: linear(vb.pp("v_proj"), d, d)?,
            o_proj: linear(vb.pp("o_proj"), d, d)?,
            ff1: linear(vb.pp("ff1"), d, cfg.d_ff)?,
            ff2: linear(vb.pp("ff2"), cfg.d_ff, d)?,
            n_heads: cfg.n_heads,
        })
    }

    /// Full-prefix causal pass. `x`: (1, T, D); `mask`: (1,1,T,T).
    pub fn forward(&self, x: &Tensor, rotary: &RotaryEmbedding, mask: &Tensor) -> Result<Tensor> {
        let normed = nn::rms_norm(x, NORM_EPS)?;
        let q = nn::split_heads(&self.q_proj.forward(&normed)?, self.n_heads)?;
        let k = nn::split_heads(&self.k_proj.forward(&normed)?, self.n_heads)?;
        let v = nn::split_heads(&self.v_proj.forward(&normed)?, self.n_heads)?;
        let (q, k) = rotary.apply_qk(&q, &k, 0)?;
        let att = nn::sdpa(&q, &k, &v, Some(mask))?;
        let x = (x + self.o_proj.forward(&nn::merge_heads(&att)?)?)?;
        let ff = self
            .ff2
            .forward(&self.ff1.forward(&nn::rms_norm(&x, NORM_EPS)?)?.gelu()?)?;
        x + ff
    }

    /// Incremental single-token pass at absolute `pos`; appends current KV to cache.
    pub fn step(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        pos: usize,
        cache: &mut Option<LayerKv>,
    ) -> Result<Tensor> {
        let normed = nn::rms_norm(x, NORM_EPS)?;
        let q = nn::split_heads(&self.q_proj.forward(&normed)?, self.n_heads)?;
        let k = nn::split_heads(&self.k_proj.forward(&normed)?, self.n_heads)?;
        let v = nn::split_heads(&self.v_proj.forward(&normed)?, self.n_heads)?;
        let (q, k) = rotary.apply_qk(&q, &k, pos)?;
        let (k_all, v_all) = match cache {
            Some(c) => {
                let app = c.appended(&k, &v)?;
                (app.k, app.v)
            }
            None => (k.clone(), v.clone()),
        };
        let att = nn::sdpa(&q, &k_all, &v_all, None)?;
        *cache = Some(LayerKv { k: k_all, v: v_all });
        let x = (x + self.o_proj.forward(&nn::merge_heads(&att)?)?)?;
        let ff = self
            .ff2
            .forward(&self.ff1.forward(&nn::rms_norm(&x, NORM_EPS)?)?.gelu()?)?;
        x + ff
    }
}

/// The Recurrent Looped Transformer model (paper §2).
pub struct Rlt {
    /// Model configuration.
    pub config: RltConfig,
    /// Variable store (owns all trainable tensors).
    pub varmap: VarMap,
    /// Device the model lives on.
    pub device: Device,
    embedding: Embedding,
    encoder: Vec<EncoderBlock>,
    decoder: Vec<DecoderBlock>,
    mem_k: Vec<Linear>,
    mem_v: Vec<Linear>,
    merge_w_g: Linear,
    merge_b_g: Tensor,
    merge_w_s: Linear,
    readout: Linear,
    s_star: Tensor,
    rotary: RotaryEmbedding,
}

impl Rlt {
    /// Create a model with fresh (randomly initialized) parameters.
    pub fn new(config: RltConfig, device: Device) -> Result<Self> {
        config.validate()?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let d = config.d_model;
        let emb = Embedding::new(
            vb.pp("embedding")
                .get((config.vocab_size, d), "embeddings")?,
            d,
        );
        let mut encoder = Vec::with_capacity(config.n_encoder_layers);
        for l in 0..config.n_encoder_layers {
            encoder.push(EncoderBlock::new(&config, vb.pp(&format!("encoder.{l}")))?);
        }
        let mut decoder = Vec::with_capacity(config.n_decoder_layers);
        for l in 0..config.n_decoder_layers {
            decoder.push(DecoderBlock::new(&config, l, &encoder, vb.pp(&format!("decoder.{l}")))?);
        }
        let mut mem_k = Vec::with_capacity(config.memory_groups);
        let mut mem_v = Vec::with_capacity(config.memory_groups);
        for g in 0..config.memory_groups {
            mem_k.push(linear(vb.pp(&format!("mem.{g}")), d, d)?);
            mem_v.push(linear(vb.pp(&format!("mem.{g}")), d, d)?);
        }
        let merge_w_g = linear(vb.pp("merge"), 2 * d, d)?;
        let merge_b_g = vb.pp("merge").get((d,), "b_g")?;
        let merge_w_s = linear(vb.pp("merge"), d, d)?;
        let readout = linear(vb.pp("readout"), d, config.vocab_size)?;
        let s_star = vb.pp("state").get((d,), "s_star")?;
        let rotary = RotaryEmbedding::new(config.head_dim(), config.max_seq_len, &device)?;
        Ok(Self {
            config,
            varmap,
            device,
            embedding: emb,
            encoder,
            decoder,
            mem_k,
            mem_v,
            merge_w_g,
            merge_b_g,
            merge_w_s,
            readout,
            s_star,
            rotary,
        })
    }

    fn embed(&self, tokens: &[u32]) -> Result<Tensor> {
        let idx = Tensor::from_vec(tokens.to_vec(), (tokens.len(),), &self.device)?;
        Ok(self.embedding.forward(&idx)?.unsqueeze(0)?) // (1, T, D)
    }

    /// Memory projections (paper Eq. 2.2) for encoder output `e`: (1, T, D).
    /// Returns per-group (k, v); the positional transform is applied to k.
    fn project_memory(&self, e: &Tensor) -> Result<Vec<GroupKv>> {
        let normed = nn::rms_norm(e, NORM_EPS)?;
        let mut out = Vec::with_capacity(self.config.memory_groups);
        for g in 0..self.config.memory_groups {
            let k = nn::split_heads(
                &self.mem_k[g].forward(&normed)?.unsqueeze(0)?,
                self.config.n_heads,
            )?;
            let v = nn::split_heads(
                &self.mem_v[g].forward(&normed)?.unsqueeze(0)?,
                self.config.n_heads,
            )?;
            let (k, _) = self.rotary.apply_qk(&k, &k, 0)?;
            out.push(GroupKv { k, v });
        }
        Ok(out)
    }

    /// Parallel causal encoding of `tokens` (paper Eq. 2.1, §2.4).
    ///
    /// Returns encoder representations (T, D), per-layer causal KV caches
    /// (full prefix), and encoder-derived global memory.
    pub fn encode(&self, tokens: &[u32]) -> Result<(Tensor, Vec<Option<LayerKv>>, Vec<GroupKv>)> {
        let t = tokens.len();
        if t == 0 || t > self.config.max_seq_len {
            return Err(RltError::State(format!(
                "encode length {t} out of range 1..={}",
                self.config.max_seq_len
            )));
        }
        let mask = nn::causal_mask(t, &self.device)?;
        let (h, caches) = self.encode_with_caches(tokens, &mask)?;
        let memory = self.project_memory(&h.unsqueeze(0)?)?;
        Ok((h, caches, memory))
    }

    /// Encoder forward that also returns post-RoPE per-layer KV caches.
    fn encode_with_caches(
        &self,
        tokens: &[u32],
        mask: &Tensor,
    ) -> Result<(Tensor, Vec<Option<LayerKv>>)> {
        let mut h = self.embed(tokens)?;
        let mut caches = Vec::with_capacity(self.encoder.len());
        for blk in &self.encoder {
            let normed = nn::rms_norm(&h, NORM_EPS)?;
            let q = nn::split_heads(&blk.q_proj.forward(&normed)?, blk.n_heads)?;
            let k = nn::split_heads(&blk.k_proj.forward(&normed)?, blk.n_heads)?;
            let v = nn::split_heads(&blk.v_proj.forward(&normed)?, blk.n_heads)?;
            let (q, k) = self.rotary.apply_qk(&q, &k, 0)?;
            let att = nn::sdpa(&q, &k, &v, Some(mask))?;
            h = (&h + blk.o_proj.forward(&nn::merge_heads(&att)?)?)?;
            let ff = blk
                .ff2
                .forward(&blk.ff1.forward(&nn::rms_norm(&h, NORM_EPS)?)?.gelu()?)?;
            h = (&h + ff)?;
            caches.push(Some(LayerKv { k, v }));
        }
        Ok((h.squeeze(0)?.contiguous()?, caches))
    }

    /// Incremental encoder step (paper Eq. 2.8): one token, updating `caches`.
    /// Returns the encoder representation e_t of shape (1, D).
    pub fn encode_step(
        &self,
        token: u32,
        pos: usize,
        caches: &mut [Option<LayerKv>],
    ) -> Result<Tensor> {
        if pos >= self.config.max_seq_len {
            return Err(RltError::State(format!(
                "position {pos} >= max_seq_len {}",
                self.config.max_seq_len
            )));
        }
        let mut x = self.embed(&[token])?; // (1,1,D)
        for (l, blk) in self.encoder.iter().enumerate() {
            x = blk.step(&x, &self.rotary, pos, &mut caches[l])?;
        }
        Ok(x.squeeze(0)?)
    }
}
```

`Rlt::new` constructs the decoder stack, so `DecoderBlock` is implemented in this
same task — append it to `rust/rlt-core/src/model.rs` (Step 3 below) so `model.rs`
is complete in one commit.

- [ ] **Step 3 (continued): append `DecoderBlock` to `rust/rlt-core/src/model.rs`**

```rust
/// One decoder layer: causal SWA self-attention → encoder-memory cross-attention →
/// FFN (paper Eq. 2.12–2.16, fixed sublayer order).
#[derive(Clone)]
pub struct DecoderBlock {
    /// SWA query projection (shares storage with encoder layer ℓ when tied).
    pub self_q: Linear,
    /// SWA key projection (shared when tied).
    pub self_k: Linear,
    /// SWA value projection (shared when tied).
    pub self_v: Linear,
    /// SWA output projection (shared when tied).
    pub self_o: Linear,
    /// Cross-attention query projection (always layer-specific).
    pub cross_q: Linear,
    /// Cross-attention output projection (always layer-specific).
    pub cross_o: Linear,
    /// FFN up projection (shared when tied).
    pub ff1: Linear,
    /// FFN down projection (shared when tied).
    pub ff2: Linear,
    /// Head count.
    pub n_heads: usize,
    /// SWA window W (includes current token).
    pub window: usize,
}

impl DecoderBlock {
    fn new(
        cfg: &RltConfig,
        layer: usize,
        encoder: &[EncoderBlock],
        vb: VarBuilder,
    ) -> Result<Self> {
        let d = cfg.d_model;
        let (self_q, self_k, self_v, self_o, ff1, ff2) = if cfg.tied {
            let e = &encoder[layer];
            (
                Linear::new(e.q_proj.weight().clone(), None),
                Linear::new(e.k_proj.weight().clone(), None),
                Linear::new(e.v_proj.weight().clone(), None),
                Linear::new(e.o_proj.weight().clone(), None),
                Linear::new(e.ff1.weight().clone(), None),
                Linear::new(e.ff2.weight().clone(), None),
            )
        } else {
            (
                linear(vb.pp("self_q"), d, d)?,
                linear(vb.pp("self_k"), d, d)?,
                linear(vb.pp("self_v"), d, d)?,
                linear(vb.pp("self_o"), d, d)?,
                linear(vb.pp("ff1"), d, cfg.d_ff)?,
                linear(vb.pp("ff2"), cfg.d_ff, d)?,
            )
        };
        Ok(Self {
            self_q,
            self_k,
            self_v,
            self_o,
            cross_q: linear(vb.pp("cross_q"), d, d)?,
            cross_o: linear(vb.pp("cross_o"), d, d)?,
            ff1,
            ff2,
            n_heads: cfg.n_heads,
            window: cfg.window,
        })
    }

    /// Single-position recurrent step at absolute `pos` (0-based), following
    /// paper Eq. 2.12–2.16. `z`: (1,1,D); `mem_prefix`: M_{≤pos} (1,H,pos+1,hd);
    /// `cache`: this layer's SWA history (≤ W−1 entries), updated in place.
    pub fn step(
        &self,
        z: &Tensor,
        rotary: &RotaryEmbedding,
        pos: usize,
        cache: &mut Option<LayerKv>,
        mem_prefix: &GroupKv,
    ) -> Result<Tensor> {
        // 1. Causal SWA self-attention with current KV (Eq. 2.12–2.14).
        let normed = nn::rms_norm(z, NORM_EPS)?;
        let q = nn::split_heads(&self.self_q.forward(&normed)?, self.n_heads)?;
        let k = nn::split_heads(&self.self_k.forward(&normed)?, self.n_heads)?;
        let v = nn::split_heads(&self.self_v.forward(&normed)?, self.n_heads)?;
        let (q, k) = rotary.apply_qk(&q, &k, pos)?;
        let (k_all, v_all) = match cache {
            Some(c) => {
                let app = c.appended(&k, &v)?;
                (app.k, app.v)
            }
            None => (k.clone(), v.clone()),
        };
        let att = nn::sdpa(&q, &k_all, &v_all, None)?;
        // Retain at most W−1 historical positions (paper §2.5 eviction convention).
        *cache = Some(LayerKv { k: k_all, v: v_all }.trimmed(self.window.saturating_sub(1))?);
        let mut z = (z + self.self_o.forward(&nn::merge_heads(&att)?)?)?;

        // 2. Encoder-memory cross-attention (Eq. 2.15). All prefix entries valid.
        let normed = nn::rms_norm(&z, NORM_EPS)?;
        let q = nn::split_heads(&self.cross_q.forward(&normed)?, self.n_heads)?;
        let q = rotary.apply(&q, pos)?;
        let att = nn::sdpa(&q, &mem_prefix.k, &mem_prefix.v, None)?;
        z = (&z + self.cross_o.forward(&nn::merge_heads(&att)?)?)?;

        // 3. Pre-norm FFN (Eq. 2.16).
        let ff = self
            .ff2
            .forward(&self.ff1.forward(&nn::rms_norm(&z, NORM_EPS)?)?.gelu()?)?;
        z + ff
    }
}
```

Also add to `impl Rlt` (same file):

```rust
    /// Gated merge of token embedding and previous recurrent output (Eq. 2.9–2.11).
    /// `e_t`: (1,1,D); `s_prev`: (1,D). Returns u_t (1,1,D).
    pub fn merge(&self, e_t: &Tensor, s_prev: &Tensor) -> Result<Tensor> {
        let r = nn::rms_norm(&s_prev.unsqueeze(0)?, NORM_EPS)?; // (1,1,D)
        let both = Tensor::cat(&[e_t, &r], D::Minus1)?; // (1,1,2D)
        let gate = candle_nn::ops::sigmoid(
            &self.merge_w_g.forward(&both)?.broadcast_add(&self.merge_b_g)?,
        )?;
        let feedback = self.merge_w_s.forward(&r)?;
        let scaled = (gate.broadcast_mul(&feedback)?).affine(self.config.feedback_alpha, 0.0)?;
        e_t + scaled
    }

    /// Readout logits (paper Eq. 2.6). `s_t`: (1,D) → (1, vocab).
    pub fn logits(&self, s_t: &Tensor) -> Result<Tensor> {
        self.readout.forward(&nn::rms_norm(s_t, NORM_EPS)?)
    }

    /// Decoder unroll for one merged token: L_D recurrent block steps.
    /// Updates `state.decoder` caches and returns the new s_t (1,D).
    pub(crate) fn decoder_unroll(&self, u: &Tensor, state: &mut crate::RltState) -> Result<Tensor> {
        let mut z = u.clone(); // (1,1,D)
        for (l, blk) in self.decoder.iter().enumerate() {
            let g = self.config.memory_group_of(l);
            let mem_len = (state.position + 1).min(state.memory[g].len());
            let mem = GroupKv {
                k: state.memory[g].k.narrow(2, 0, mem_len)?,
                v: state.memory[g].v.narrow(2, 0, mem_len)?,
            };
            z = blk.step(&z, &self.rotary, state.position, &mut state.decoder[l], &mem)?;
        }
        z.squeeze(0)
    }
```

- [ ] **Step 4: Run the encoder test** — `cd rust && cargo test -p rlt-core --test encoder 2>&1 | tail -5`; expect `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add rust/rlt-core/src/model.rs rust/rlt-core/src/lib.rs rust/rlt-core/tests/encoder.rs
git commit -m "rust(rlt-core): full model — encoder, decoder, merge, readout, tied weights"
```

---

### Task 8: `execution.rs` — `initial_state`, `prefill`, `step`

**Files:**
- Create: `rust/rlt-core/src/execution.rs`
- Modify: `rust/rlt-core/src/lib.rs` — add `pub mod execution;` and append a doctest
  to the crate docs (below)
- Test: `rust/rlt-core/tests/execution.rs`

Add this doctest to `lib.rs` crate docs (after the existing paragraph):

```rust
//! ```
//! use rlt_core::{Rlt, RltConfig, ByteTokenizer, Device};
//! # fn main() -> rlt_core::Result<()> {
//! let model = Rlt::new(RltConfig::default(), Device::Cpu)?;
//! let tok = ByteTokenizer::new(258)?;
//! let (logits, state) = model.prefill(&tok.encode("Hi", true, false))?;
//! assert_eq!(logits.len(), 3);
//! # Ok(()) }
//! ```
```

- [ ] **Step 1: Write the failing execution tests (the paper's core invariants)**

```rust
use candle_core::Device;
use rlt_core::{RltConfig, Rlt};

fn tiny_config() -> RltConfig {
    RltConfig {
        d_model: 32, n_heads: 2, d_ff: 64,
        n_encoder_layers: 2, n_decoder_layers: 2,
        window: 3, memory_groups: 1, vocab_size: 258,
        tied: false, feedback_alpha: 0.1, max_seq_len: 64,
    }
}

fn max_diff(a: &candle_core::Tensor, b: &candle_core::Tensor) -> f32 {
    (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap()
}

/// Paper Proposition 3.1: prefill ≡ incremental step-by-step execution.
#[test]
fn prefill_matches_incremental() {
    let model = Rlt::new(tiny_config(), Device::Cpu).unwrap();
    let tokens: Vec<u32> = vec![7, 1, 3, 3, 5, 2, 8, 4, 9, 0, 6, 11];
    let (prefill_logits, prefill_state) = model.prefill(&tokens).unwrap();
    let mut state = model.initial_state().unwrap();
    for (i, &t) in tokens.iter().enumerate() {
        let logits = model.step(t, &mut state).unwrap();
        let d = max_diff(&logits, &prefill_logits[i]);
        assert!(d < 1e-4, "position {i}: prefill vs step diff {d}");
    }
    assert_eq!(state.position, tokens.len());
    let d = max_diff(&state.s, &prefill_state.s);
    assert!(d < 1e-4, "final recurrent state diff {d}");
}

/// Paper Proposition B.1: position j logits depend only on tokens ≤ j.
#[test]
fn causality_holds() {
    let model = Rlt::new(tiny_config(), Device::Cpu).unwrap();
    let a: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let mut b = a.clone();
    b[5] = 200; // mutate a later token only
    let (la, _) = model.prefill(&a).unwrap();
    let (lb, _) = model.prefill(&b).unwrap();
    for i in 0..=4 {
        let d = max_diff(&la[i], &lb[i]);
        assert!(d < 1e-6, "logits at {i} changed by future token: {d}");
    }
    assert!(max_diff(&la[7], &lb[7]) > 1e-4);
}

/// SWA cache semantics: after n tokens each decoder cache holds exactly
/// min(W−1, n) positions; W=1 keeps caches permanently empty.
#[test]
fn swa_cache_lengths() {
    let cfg = tiny_config();
    let model = Rlt::new(cfg, Device::Cpu).unwrap();
    let (_, state) = model.prefill(&[1, 2, 3, 4, 5]).unwrap();
    for c in state.decoder.iter().flatten() {
        assert_eq!(c.len(), cfg.window - 1, "cache should hold W-1 entries");
    }
    let mut w1 = tiny_config();
    w1.window = 1;
    let model1 = Rlt::new(w1, Device::Cpu).unwrap();
    let (lg, st1) = model1.prefill(&[1, 2, 3]).unwrap();
    assert!(st1.decoder.iter().all(|c| c.is_none()));
    assert_eq!(lg.len(), 3);
}

/// Prompt–response continuity: step after prefill matches full-sequence prefill.
#[test]
fn generation_continues_state() {
    let model = Rlt::new(tiny_config(), Device::Cpu).unwrap();
    let prompt = vec![9u32, 8, 7];
    let (_, mut state) = model.prefill(&prompt).unwrap();
    assert_eq!(state.position, prompt.len());
    let next = model.step(42, &mut state).unwrap();
    let (full, _) = model.prefill(&[9, 8, 7, 42]).unwrap();
    let d2 = max_diff(&next, &full[3]);
    assert!(d2 < 1e-4, "step after prefill diverged from full prefill: {d2}");
}
```

- [ ] **Step 2: Run to verify failure** — `prefill`, `initial_state`, `step` missing.

- [ ] **Step 3: Implement `rust/rlt-core/src/execution.rs`**

```rust
//! Execution paths (paper §2.3–2.5, App. A): prefill, incremental step, training.

use candle_core::{D, Result, Tensor};
use candle_nn::{AdamW, Optimizer};

use crate::model::Rlt;
use crate::state::{GroupKv, RltState};
use crate::RltError;

impl Rlt {
    /// Initial state H_0 = (s_star, ∅) (paper Eq. 2.3).
    pub fn initial_state(&self) -> Result<RltState> {
        Ok(RltState {
            s: self.s_star.clone().unsqueeze(0)?, // (1, D)
            decoder: vec![None; self.config.n_decoder_layers],
            encoder: vec![None; self.config.n_encoder_layers],
            memory: Vec::new(),
            position: 0,
        })
    }

    /// Prompt prefill (paper §A.1): parallel causal encoder pass, then the decoder
    /// transition for EVERY prompt token in order. Returns per-position logits and
    /// the complete state H_T.
    pub fn prefill(&self, tokens: &[u32]) -> Result<(Vec<Tensor>, RltState)> {
        let (e, enc_caches, memory) = self.encode(tokens)?; // (T, D)
        let mut state = self.initial_state()?;
        state.encoder = enc_caches;
        state.memory = memory;
        let mut logits = Vec::with_capacity(tokens.len());
        for i in 0..tokens.len() {
            if state.position >= self.config.max_seq_len {
                return Err(RltError::State("max_seq_len exceeded during prefill".into()));
            }
            let e_t = e.narrow(0, i, 1)?.unsqueeze(0)?; // (1,1,D)
            let u = self.merge(&e_t, &state.s)?;
            state.s = self.decoder_unroll(&u, &mut state)?; // (1,D)
            state.position += 1;
            logits.push(self.logits(&state.s)?); // (1,V)
        }
        Ok((logits, state))
    }

    /// Incremental consumption of one token (paper §A.2, Eq. 2.8 + transition):
    /// encoder step, memory append, merge, decoder unroll, readout.
    /// Returns logits (1, vocab) for the NEXT token.
    pub fn step(&self, token: u32, state: &mut RltState) -> Result<Tensor> {
        if state.position >= self.config.max_seq_len {
            return Err(RltError::State("max_seq_len exceeded".into()));
        }
        let pos = state.position;
        let e_t = self.encode_step(token, pos, &mut state.encoder)?; // (1,D)
        let e_row = e_t.unsqueeze(0)?; // (1,1,D)
        self.append_memory(&e_row, pos, state)?;
        let u = self.merge(&e_row, &state.s)?;
        state.s = self.decoder_unroll(&u, state)?;
        state.position = pos + 1;
        self.logits(&state.s)
    }

    /// Append the current position to encoder-derived memory (Eq. 2.2).
    fn append_memory(&self, e_row: &Tensor, pos: usize, state: &mut RltState) -> Result<()> {
        let g_proj = |g: usize| -> Result<(Tensor, Tensor)> {
            let normed = crate::nn::rms_norm(e_row, crate::nn::NORM_EPS)?;
            let k = crate::nn::split_heads(
                &self.mem_k[g].forward(&normed)?,
                self.config.n_heads,
            )?;
            let v = crate::nn::split_heads(
                &self.mem_v[g].forward(&normed)?,
                self.config.n_heads,
            )?;
            let (k, _) = self.rotary.apply_qk(&k, &k, pos)?;
            Ok((k, v))
        };
        if state.memory.is_empty() {
            for g in 0..self.config.memory_groups {
                let (k, v) = g_proj(g)?;
                state.memory.push(GroupKv { k, v });
            }
            return Ok(());
        }
        if state.memory[0].len() != pos {
            return Err(RltError::State(format!(
                "memory/state desync: memory has {} entries, position {pos}",
                state.memory[0].len()
            )));
        }
        for g in 0..self.config.memory_groups {
            let (k, v) = g_proj(g)?;
            state.memory[g] = state.memory[g].appended(&k, &v)?;
        }
        Ok(())
    }

    /// Differentiable masked-mean next-token loss over an unrolled sequence
    /// (paper §5.1–5.2). Full BPTT: nothing is detached. `loss_mask` (length
    /// `tokens.len() - 1`, 1.0 = supervise this target, 0.0 = state-only token).
    pub fn forward_loss(&self, tokens: &[u32], loss_mask: Option<&[f32]>) -> Result<Tensor> {
        let n = tokens.len();
        if n < 2 {
            return Err(RltError::State("forward_loss needs at least 2 tokens".into()));
        }
        let (logits, _) = self.prefill(tokens)?;
        let stacked = Tensor::stack(
            &logits[..n - 1]
                .iter()
                .map(|l| l.squeeze(0))
                .collect::<Result<Vec<_>>>()?,
            0,
        )?; // (T-1, V)
        let logp = candle_nn::ops::log_softmax(&stacked, D::Minus1)?;
        let targets = Tensor::from_vec(tokens[1..].to_vec(), (n - 1, 1), &self.device)?;
        let nll = logp.gather(&targets, 1)?.squeeze(1)?.affine(-1.0, 0.0)?; // (T-1,)
        let mask = match loss_mask {
            Some(m) => {
                if m.len() != n - 1 {
                    return Err(RltError::State(format!(
                        "loss_mask length {} != {}",
                        m.len(),
                        n - 1
                    )));
                }
                if m.iter().all(|&x| x == 0.0) {
                    return Err(RltError::State("loss_mask has no supervised targets".into()));
                }
                Tensor::from_vec(m.to_vec(), (n - 1,), &self.device)?
            }
            None => Tensor::ones((n - 1,), candle_core::DType::F32, &self.device)?,
        };
        let num = nll.broadcast_mul(&mask)?.sum_all()?;
        let den = mask.sum_all()?;
        num.broadcast_div(&den)
    }

    /// One full-BPTT optimizer step; returns the scalar loss.
    pub fn train_step(
        &self,
        tokens: &[u32],
        loss_mask: Option<&[f32]>,
        opt: &mut AdamW,
    ) -> Result<f64> {
        let loss = self.forward_loss(tokens, loss_mask)?;
        opt.backward_step(&loss)?;
        Ok(loss.to_scalar::<f32>()? as f64)
    }
}
```

- [ ] **Step 4: Run tests** — `cd rust && cargo test -p rlt-core --test execution 2>&1 | tail -5` — expect `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add rust/rlt-core/src/execution.rs rust/rlt-core/src/lib.rs rust/rlt-core/tests/execution.rs
git commit -m "rust(rlt-core): prefill, incremental step, full-BPTT train_step"
```

---

### Task 9: Gradient and masking verification

**Files:**
- Test: `rust/rlt-core/tests/gradients.rs`

- [ ] **Step 1: Write the tests**

```rust
use candle_core::{Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rlt_core::{RltConfig, Rlt};

fn tiny_config() -> RltConfig {
    RltConfig {
        d_model: 16, n_heads: 2, d_ff: 32,
        n_encoder_layers: 1, n_decoder_layers: 1,
        window: 2, memory_groups: 1, vocab_size: 258,
        tied: false, feedback_alpha: 0.1, max_seq_len: 32,
    }
}

/// 4-D extraction (candle has no `to_vec4`): reshape to 3-D and regroup.
fn t4(t: &Tensor) -> Vec<Vec<Vec<Vec<f32>>>> {
    let (a, b, c, d) = t.dims4().unwrap();
    t.reshape((a * b, c, d)).unwrap()
        .to_vec3::<f32>().unwrap()
        .chunks(b)
        .map(|c| c.to_vec())
        .collect()
}

/// Finite-difference check of candle backward through rms_norm + sdpa + rope.
#[test]
fn fd_matches_backprop_on_attention_ops() {
    let dev = Device::Cpu;
    let x = Var::from_tensor(&Tensor::rand(-0.5f32, 0.5, (1, 2, 3, 8), &dev).unwrap()).unwrap();
    let w = Var::from_tensor(&Tensor::rand(-0.5f32, 0.5, (8, 8), &dev).unwrap()).unwrap();
    let rotary = rlt_core::nn::RotaryEmbedding::new(4, 8, &dev).unwrap();
    let loss_fn = |x: &Tensor, w: &Tensor| -> Tensor {
        let xw = x.matmul(w).unwrap();
        let q = rlt_core::nn::split_heads(&xw.squeeze(0).unwrap(), 2).unwrap(); // (2,3,8)
        let q = q.unsqueeze(0).unwrap(); // (1,2,3,8)
        let (q, k) = rotary.apply_qk(&q, &q, 0).unwrap();
        let att = rlt_core::nn::sdpa(&q, &k, &q, None).unwrap();
        rlt_core::nn::rms_norm(&rlt_core::nn::merge_heads(&att).unwrap().squeeze(0).unwrap(), 1e-5)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
    };
    let loss = loss_fn(x.as_tensor(), w.as_tensor());
    let grads = loss.backward().unwrap();
    let analytic_x = grads.get(x.as_tensor()).unwrap().clone();
    let analytic_w = grads.get(w.as_tensor()).unwrap().clone();
    let eps = 1e-3f32;
    let mut xv = t4(x.as_tensor());
    for idx in [(0usize, 0usize, 0usize, 0usize), (0, 1, 2, 7)] {
        let orig = xv[idx.0][idx.1][idx.2][idx.3];
        xv[idx.0][idx.1][idx.2][idx.3] = orig + eps;
        let lp = loss_fn(&Tensor::new(xv.clone(), &dev).unwrap(), w.as_tensor());
        xv[idx.0][idx.1][idx.2][idx.3] = orig - eps;
        let lm = loss_fn(&Tensor::new(xv.clone(), &dev).unwrap(), w.as_tensor());
        xv[idx.0][idx.1][idx.2][idx.3] = orig;
        let fd = (lp.to_scalar::<f32>().unwrap() - lm.to_scalar::<f32>().unwrap()) / (2.0 * eps);
        let an = t4(&analytic_x)[idx.0][idx.1][idx.2][idx.3];
        assert!((fd - an).abs() < 5e-3, "x{idx:?}: fd={fd} analytic={an}");
    }
    let mut wv = w.as_tensor().to_vec2::<f32>().unwrap();
    for idx in [(0usize, 0usize), (7, 7)] {
        let orig = wv[idx.0][idx.1];
        wv[idx.0][idx.1] = orig + eps;
        let lp = loss_fn(x.as_tensor(), &Tensor::new(wv.clone(), &dev).unwrap());
        wv[idx.0][idx.1] = orig - eps;
        let lm = loss_fn(x.as_tensor(), &Tensor::new(wv.clone(), &dev).unwrap());
        wv[idx.0][idx.1] = orig;
        let fd = (lp.to_scalar::<f32>().unwrap() - lm.to_scalar::<f32>().unwrap()) / (2.0 * eps);
        let an = analytic_w.to_vec2::<f32>().unwrap()[idx.0][idx.1];
        assert!((fd - an).abs() < 5e-3, "w{idx:?}: fd={fd} analytic={an}");
    }
}

/// Finite-difference check on real model parameters (merge bias, encoder
/// attention projection) through the full recurrent unroll.
#[test]
fn fd_matches_backprop_on_named_parameters() {
    let model = Rlt::new(tiny_config(), Device::Cpu).unwrap();
    let tokens: Vec<u32> = vec![1, 2, 3, 4];
    let eps = 1e-3f32;
    for name in ["merge.b_g", "encoder.0.q_proj.weight"] {
        let loss = model.forward_loss(&tokens, None).unwrap();
        let grads = loss.backward().unwrap();
        let var = model.varmap.data().lock().unwrap().get(name).unwrap().clone();
        let t = var.as_tensor();
        let shape: Vec<usize> = t.dims().to_vec();
        let flat = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let analytic = grads
            .get(t)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for idx in [0usize, flat.len() / 2] {
            let orig = flat[idx];
            let mut plus = flat.clone();
            plus[idx] = orig + eps;
            var.set(&Tensor::from_vec(plus, shape.clone(), &Device::Cpu).unwrap())
                .unwrap();
            let lp = model.forward_loss(&tokens, None).unwrap().to_scalar::<f32>().unwrap();
            let mut minus = flat.clone();
            minus[idx] = orig - eps;
            var.set(&Tensor::from_vec(minus, shape.clone(), &Device::Cpu).unwrap())
                .unwrap();
            let lm = model.forward_loss(&tokens, None).unwrap().to_scalar::<f32>().unwrap();
            var.set(&Tensor::from_vec(flat.clone(), shape.clone(), &Device::Cpu).unwrap())
                .unwrap();
            let fd = (lp - lm) / (2.0 * eps);
            assert!(
                (fd - analytic[idx]).abs() < 5e-3,
                "{name}[{idx}]: fd={fd} analytic={}",
                analytic[idx]
            );
        }
    }
}

/// Every trainable variable receives a gradient through the full recurrent unroll.
#[test]
fn backward_reaches_all_parameters() {
    let model = Rlt::new(tiny_config(), Device::Cpu).unwrap();
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
    let loss = model.forward_loss(&tokens, None).unwrap();
    let grads = loss.backward().unwrap();
    let vars = model.varmap.all_vars();
    assert!(vars.len() > 10);
    let missing_count = vars.iter().filter(|v| grads.get(v.as_tensor()).is_none()).count();
    assert_eq!(missing_count, 0, "some parameters got no gradient");
}

/// SFT masking: loss changes; the unmasked forward computation is unaffected
/// (masks only reweight the loss — state updates always run, paper §5.2).
#[test]
fn masking_reweights_only_the_loss() {
    let model = Rlt::new(tiny_config(), Device::Cpu).unwrap();
    let tokens: Vec<u32> = vec![10, 11, 12, 13, 14];
    let full = model.forward_loss(&tokens, None).unwrap().to_scalar::<f32>().unwrap();
    let masked = model
        .forward_loss(&tokens, Some(&[0.0, 0.0, 0.0, 1.0]))
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!((masked - full).abs() > 1e-6, "mask had no effect");
    assert!(masked > 0.0 && masked.is_finite());
}

/// Full-BPTT training reduces loss on a trivially predictable stream.
#[test]
fn train_step_decreases_loss() {
    let model = Rlt::new(tiny_config(), Device::Cpu).unwrap();
    let mut opt = AdamW::new(
        model.varmap.all_vars(),
        ParamsAdamW { lr: 0.01, ..Default::default() },
    )
    .unwrap();
    let tokens: Vec<u32> = vec![42u32; 12];
    let first = model.train_step(&tokens, None, &mut opt).unwrap();
    let mut last = first;
    for _ in 0..29 {
        last = model.train_step(&tokens, None, &mut opt).unwrap();
    }
    assert!(last < first * 0.9, "loss did not decrease: {first} -> {last}");
}
```

- [ ] **Step 2: Run tests** — `cd rust && cargo test -p rlt-core --test gradients 2>&1 | tail -8`
Expected: `5 passed`. The FD tolerances (5e-3) may be loosened to 2e-2 if the loss
landscape is sharp; never beyond 5e-2 without investigating a real bug.

- [ ] **Step 3: Commit**

```bash
git add rust/rlt-core/tests/gradients.rs
git commit -m "rust(rlt-core): gradient FD checks, full-param grad reachability, masking, train smoke"
```

---

### Task 10: `sampling.rs` + `generate`

**Files:**
- Create: `rust/rlt-core/src/sampling.rs`
- Modify: `rust/rlt-core/src/execution.rs` — add `generate`
- Modify: `rust/rlt-core/src/lib.rs` — add `pub mod sampling;` and
  `pub use sampling::{sample_from_logits, SampledToken, Sampler};`
- Test: `rust/rlt-core/tests/sampling.rs`

- [ ] **Step 1: Write the failing tests**

```rust
use candle_core::{D, Device, Tensor};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rlt_core::{sample_from_logits, Rlt, RltConfig, Sampler};

fn logits_of(v: Vec<f32>) -> Tensor {
    Tensor::new(v, &Device::Cpu).unwrap().unsqueeze(0).unwrap() // (1,V)
}

#[test]
fn greedy_picks_argmax_with_correct_logprob() {
    let logits = logits_of(vec![1.0, 3.0, 2.0, 0.5]);
    let mut rng = StdRng::seed_from_u64(0);
    let tok = sample_from_logits(&logits, &Sampler::Greedy, &mut rng).unwrap();
    assert_eq!(tok.token, 1);
    let lp = logits.log_softmax(D::Minus1).unwrap().to_vec2::<f32>().unwrap()[0][1];
    assert!((tok.logprob - lp).abs() < 1e-5);
}

#[test]
fn seeded_sampling_is_deterministic() {
    let logits = logits_of(vec![2.0, 1.0, 0.5, 3.0]);
    let mut r1 = StdRng::seed_from_u64(42);
    let mut r2 = StdRng::seed_from_u64(42);
    let s = Sampler::Sample { temperature: 0.8, top_k: Some(2) };
    let a = sample_from_logits(&logits, &s, &mut r1).unwrap();
    let b = sample_from_logits(&logits, &s, &mut r2).unwrap();
    assert_eq!(a.token, b.token);
}

#[test]
fn temperature_zero_is_rejected() {
    let logits = logits_of(vec![1.0, 2.0]);
    let mut rng = StdRng::seed_from_u64(0);
    let s = Sampler::Sample { temperature: 0.0, top_k: None };
    assert!(sample_from_logits(&logits, &s, &mut rng).is_err());
}

#[test]
fn generate_runs_and_stops_at_eos() {
    let cfg = RltConfig {
        d_model: 32, n_heads: 2, d_ff: 64,
        n_encoder_layers: 2, n_decoder_layers: 2,
        window: 4, memory_groups: 1, vocab_size: 258,
        tied: false, feedback_alpha: 0.1, max_seq_len: 64,
    };
    let model = Rlt::new(cfg, Device::Cpu).unwrap();
    let mut rng = StdRng::seed_from_u64(1);
    let (tokens, state) = model
        .generate(&[1, 2, 3], 16, &Sampler::Greedy, &mut rng, true)
        .unwrap();
    assert!(!tokens.is_empty());
    assert!(tokens.len() <= 16);
    assert_eq!(state.position, 3 + tokens.len());
    if tokens.last().unwrap().token == rlt_core::EOS_ID {
        assert!(tokens.len() < 16, "should stop at EOS");
    }
}
```

- [ ] **Step 2: Run to verify failure** — `sample_from_logits`, `Sampler`, `generate`
missing.

- [ ] **Step 3: Implement `rust/rlt-core/src/sampling.rs`**

```rust
//! Sampling from readout logits: greedy, temperature, top-k.

use candle_core::{Result, Tensor};
use rand::rngs::StdRng;
use rand::Rng;

use crate::RltError;

/// Sampling strategy for one token.
#[derive(Clone, Copy, Debug)]
pub enum Sampler {
    /// Deterministic argmax.
    Greedy,
    /// Stochastic sampling with temperature and optional top-k truncation
    /// (the truncated distribution is renormalized, matching replay requirements).
    Sample {
        /// Temperature (> 0); divides logits before softmax.
        temperature: f32,
        /// Optional top-k truncation.
        top_k: Option<usize>,
    },
}

/// A sampled token and its log-probability under the model's full-vocab
/// distribution — the value RL rollouts must record as the behavior log-probability
/// when sampling untruncated targets (paper §5.3).
#[derive(Clone, Debug)]
pub struct SampledToken {
    /// Sampled token id.
    pub token: u32,
    /// log-probability of this token under the model distribution.
    pub logprob: f32,
}

/// Sample one token from logits (1, vocab).
pub fn sample_from_logits(
    logits: &Tensor,
    sampler: &Sampler,
    rng: &mut StdRng,
) -> Result<SampledToken> {
    let v = logits.squeeze(0)?.to_vec1::<f32>()?;
    let logp_full = log_softmax_vec(&v);
    match sampler {
        Sampler::Greedy => {
            let (idx, _) = v
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .ok_or_else(|| RltError::State("empty logits".into()))?;
            Ok(SampledToken { token: idx as u32, logprob: logp_full[idx] })
        }
        Sampler::Sample { temperature, top_k } => {
            if *temperature <= 0.0 {
                return Err(RltError::Config("sampling temperature must be > 0".into()));
            }
            let mut cand: Vec<(u32, f32)> = v
                .iter()
                .enumerate()
                .map(|(i, &x)| (i as u32, x / temperature))
                .collect();
            if let Some(k) = top_k {
                cand.sort_by(|a, b| b.1.total_cmp(&a.1));
                cand.truncate(k.max(1));
            }
            let probs = softmax_vec(&cand.iter().map(|&(_, x)| x).collect::<Vec<_>>());
            let r = rng.random::<f64>();
            let mut acc = 0f64;
            for (j, &(tok, _)) in cand.iter().enumerate() {
                acc += f64::from(probs[j]);
                if r <= acc {
                    return Ok(SampledToken { token: tok, logprob: logp_full[tok as usize] });
                }
            }
            let (tok, _) = cand[cand.len() - 1];
            Ok(SampledToken { token: tok, logprob: logp_full[tok as usize] })
        }
    }
}

fn softmax_vec(xs: &[f32]) -> Vec<f32> {
    let m = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| (x - m).exp()).collect();
    let z: f32 = exps.iter().sum();
    exps.iter().map(|e| e / z).collect()
}

fn log_softmax_vec(xs: &[f32]) -> Vec<f32> {
    let m = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let z: f32 = xs.iter().map(|x| (x - m).exp()).sum();
    xs.iter().map(|x| x - m - z.ln()).collect()
}
```

- [ ] **Step 4: Add `generate` to `rust/rlt-core/src/execution.rs`**

```rust
    /// Autoregressive generation (paper §A.2): prefill the prompt, then sample and
    /// consume tokens. Returns sampled tokens (with log-probs) and the final state.
    pub fn generate(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        sampler: &crate::Sampler,
        rng: &mut rand::rngs::StdRng,
        stop_at_eos: bool,
    ) -> Result<(Vec<crate::SampledToken>, RltState)> {
        let (_, mut state) = self.prefill(prompt)?;
        let mut out = Vec::new();
        for _ in 0..max_tokens {
            let logits = self.logits(&state.s)?;
            let tok = crate::sample_from_logits(&logits, sampler, rng)?;
            let is_eos = tok.token == crate::tokenizer::EOS_ID;
            out.push(tok.clone());
            self.step(tok.token, &mut state)?;
            if stop_at_eos && is_eos {
                break;
            }
        }
        Ok((out, state))
    }
```

- [ ] **Step 5: Run tests** — `cd rust && cargo test -p rlt-core --test sampling 2>&1 | tail -5`; expect `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add rust/rlt-core/src/sampling.rs rust/rlt-core/src/execution.rs rust/rlt-core/src/lib.rs rust/rlt-core/tests/sampling.rs
git commit -m "rust(rlt-core): sampling + autoregressive generate"
```

---

### Task 11: `replay.rs` — current-policy RL replay

**Files:**
- Create: `rust/rlt-core/src/replay.rs`
- Modify: `rust/rlt-core/src/lib.rs` — add `pub mod replay;` and
  `pub use replay::{replay, ReplayResult, Rollout, SamplingMetadata};`
- Test: `rust/rlt-core/tests/replay.rs`

- [ ] **Step 1: Write the failing test**

```rust
use candle_core::Device;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rlt_core::{replay, RltConfig, Rollout, Sampler};

fn tiny_config() -> RltConfig {
    RltConfig {
        d_model: 32, n_heads: 2, d_ff: 64,
        n_encoder_layers: 2, n_decoder_layers: 2,
        window: 3, memory_groups: 1, vocab_size: 258,
        tied: false, feedback_alpha: 0.1, max_seq_len: 64,
    }
}

#[test]
fn replay_ratios_are_one_at_same_params() {
    let model = Rlt::new(tiny_config(), Device::Cpu).unwrap();
    let prompt = vec![5u32, 4, 3, 2, 1];
    let mut rng = StdRng::seed_from_u64(7);
    let (sampled, _) = model
        .generate(&prompt, 6, &Sampler::Greedy, &mut rng, false)
        .unwrap();
    let response: Vec<u32> = sampled.iter().map(|t| t.token).collect();
    let behavior: Vec<f32> = sampled.iter().map(|t| t.logprob).collect();
    let mut mask = vec![true; response.len()];
    mask[2] = false; // external token at index 2 is not a policy action
    let rollout = Rollout {
        prompt_tokens: prompt.clone(),
        response_tokens: response.clone(),
        action_mask: mask.clone(),
        behavior_logprobs: behavior.clone(),
        sampling: rlt_core::SamplingMetadata { temperature: 0.0, top_k: None, seed: 7 },
    };
    let result = replay(&model, &rollout).unwrap();
    assert_eq!(result.ratios.len(), mask.iter().filter(|&&m| m).count());
    for r in &result.ratios {
        assert!((r - 1.0).abs() < 1e-4, "ratio {r} != 1 at identical params");
    }
    let cur = result.current_logprobs.to_vec1::<f32>().unwrap();
    let mut ci = 0;
    for (i, &m) in mask.iter().enumerate() {
        if m {
            assert!((cur[ci] - behavior[i]).abs() < 1e-4);
            ci += 1;
        }
    }
}

#[test]
fn replay_rejects_bad_shapes() {
    let model = Rlt::new(tiny_config(), Device::Cpu).unwrap();
    let rollout = Rollout {
        prompt_tokens: vec![1, 2],
        response_tokens: vec![3, 4],
        action_mask: vec![true],
        behavior_logprobs: vec![0.0, 0.0],
        sampling: rlt_core::SamplingMetadata { temperature: 1.0, top_k: None, seed: 0 },
    };
    assert!(replay(&model, &rollout).is_err());
}
```

- [ ] **Step 2: Run to verify failure** — `replay`, `Rollout`, `SamplingMetadata`
missing.

- [ ] **Step 3: Implement `rust/rlt-core/src/replay.rs`**

```rust
//! Current-policy RL replay (paper §5.3, §A.4).
//!
//! The sampler records tokens, an action mask, and behavior log-probabilities under
//! the actual sampling distribution µ (including any temperature/top-k transforms).
//! The trainer replays the complete history under current parameters Θ — full
//! prompt recurrence, no stale caches, no detach — and forms token ratios
//! `r_i = exp(log p_Θ − log µ)` for policy actions only.

use candle_core::{D, Result, Tensor};

use crate::model::Rlt;
use crate::RltError;

/// Record of how the response was sampled (metadata only; behavior log-probs carry
/// the actual probabilities).
#[derive(Clone, Debug)]
pub struct SamplingMetadata {
    /// Sampling temperature (0 = greedy).
    pub temperature: f32,
    /// Optional top-k truncation.
    pub top_k: Option<usize>,
    /// Sampler RNG seed.
    pub seed: u64,
}

/// A recorded rollout: prompt plus sampled response with behavior log-probs.
#[derive(Clone, Debug)]
pub struct Rollout {
    /// Observed prompt tokens (including BOS as recorded).
    pub prompt_tokens: Vec<u32>,
    /// Sampled response tokens.
    pub response_tokens: Vec<u32>,
    /// Per-response-token policy-action flags (false = external user/tool token).
    pub action_mask: Vec<bool>,
    /// Behavior log-probabilities log µ(y_i | ·) as recorded by the sampler.
    pub behavior_logprobs: Vec<f32>,
    /// Sampling metadata for the record.
    pub sampling: SamplingMetadata,
}

/// Result of replaying a rollout under current parameters.
#[derive(Debug)]
pub struct ReplayResult {
    /// Stacked current-policy log-probabilities log p_Θ(y_i | ·) for ACTION tokens
    /// (shape (A,)); the autograd graph spans the full prompt+response replay.
    pub current_logprobs: Tensor,
    /// Importance ratios r_i = exp(log p_Θ − log µ), one per action token.
    pub ratios: Vec<f32>,
}

/// Replay a rollout under the model's current parameters (paper §A.4).
///
/// Every token — prompt and response, action or external — is consumed exactly
/// once; actions are scored from the preceding state before being consumed.
pub fn replay(model: &Rlt, rollout: &Rollout) -> Result<ReplayResult> {
    let n = rollout.response_tokens.len();
    if rollout.action_mask.len() != n || rollout.behavior_logprobs.len() != n {
        return Err(RltError::Replay(format!(
            "response/mask/logprob length mismatch: {n} vs {} vs {}",
            rollout.action_mask.len(),
            rollout.behavior_logprobs.len()
        )));
    }
    let (_, mut state) = model.prefill(&rollout.prompt_tokens)?;
    let mut cur_logp: Vec<Tensor> = Vec::new();
    let mut ratios: Vec<f32> = Vec::new();
    for (i, &tok) in rollout.response_tokens.iter().enumerate() {
        let logits = model.logits(&state.s)?; // (1,V)
        let lp = candle_nn::ops::log_softmax(&logits, D::Minus1)?
            .gather(&Tensor::new(&[tok], &model.device)?.unsqueeze(1)?, 1)?
            .squeeze(1)?; // (1,)
        if rollout.action_mask[i] {
            let cur = lp.to_scalar::<f32>()?;
            ratios.push((cur - rollout.behavior_logprobs[i]).exp());
            cur_logp.push(lp);
        }
        model.step(tok, &mut state)?;
    }
    if cur_logp.is_empty() {
        return Err(RltError::Replay("rollout has no action tokens".into()));
    }
    Ok(ReplayResult { current_logprobs: Tensor::stack(&cur_logp, 0)?, ratios })
}
```

- [ ] **Step 4: Run tests** — `cd rust && cargo test -p rlt-core --test replay 2>&1 | tail -5`; expect `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add rust/rlt-core/src/replay.rs rust/rlt-core/src/lib.rs rust/rlt-core/tests/replay.rs
git commit -m "rust(rlt-core): current-policy RL replay with ratios"
```

---

### Task 12: `checkpoint.rs`

**Files:**
- Create: `rust/rlt-core/src/checkpoint.rs`
- Modify: `rust/rlt-core/src/lib.rs` — add `pub mod checkpoint;` and
  `pub use checkpoint::{load_checkpoint, save_checkpoint};`
- Test: `rust/rlt-core/tests/checkpoint.rs`

- [ ] **Step 1: Write the failing test**

```rust
use candle_core::Device;
use rlt_core::{load_checkpoint, save_checkpoint, Rlt, RltConfig};

fn tiny_config() -> RltConfig {
    RltConfig {
        d_model: 32, n_heads: 2, d_ff: 64,
        n_encoder_layers: 2, n_decoder_layers: 2,
        window: 4, memory_groups: 1, vocab_size: 258,
        tied: false, feedback_alpha: 0.1, max_seq_len: 64,
    }
}

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("rlt_test_{}_{}.st", name, std::process::id()))
}

#[test]
fn checkpoint_roundtrip_preserves_logits() {
    let path = tmp("roundtrip");
    let cfg = tiny_config();
    let model = Rlt::new(cfg.clone(), Device::Cpu).unwrap();
    let tokens = vec![3u32, 1, 4, 1, 5, 9];
    let (before, _) = model.prefill(&tokens).unwrap();
    save_checkpoint(&model, &path).unwrap();
    let loaded = load_checkpoint(&path, Device::Cpu).unwrap();
    assert_eq!(loaded.config, cfg);
    let (after, _) = loaded.prefill(&tokens).unwrap();
    for (a, b) in before.iter().zip(after.iter()) {
        let d = (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(d, 0.0, "logits changed across checkpoint");
    }
    std::fs::remove_file(&path).ok();
    std::fs::remove_file(path.with_extension("st.json")).ok();
}

#[test]
fn load_missing_config_errors() {
    let path = tmp("missing");
    assert!(load_checkpoint(&path, Device::Cpu).is_err());
}
```

- [ ] **Step 2: Run to verify failure** — `save_checkpoint`/`load_checkpoint` missing.

- [ ] **Step 3: Implement `rust/rlt-core/src/checkpoint.rs`**

```rust
//! Checkpointing: safetensors weights (via candle's VarMap) + JSON config sidecar.
//!
//! A checkpoint is `PATH` (safetensors weights) plus `PATH` with extension
//! `st.json` (model configuration). Optimizer state is not persisted: resuming
//! training restarts the optimizer (warm restart), a documented v1 simplification.

use std::path::{Path, PathBuf};

use candle_core::Device;
use candle_nn::VarMap;

use crate::{Rlt, RltConfig, RltError, Result};

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
        return Err(RltError::Checkpoint(format!("no checkpoint at {}", path.display())));
    }
    let cfg_path = config_path(path);
    let json = std::fs::read_to_string(&cfg_path)
        .map_err(|e| RltError::Checkpoint(format!("cannot read {}: {e}", cfg_path.display())))?;
    let config: RltConfig =
        serde_json::from_str(&json).map_err(|e| RltError::Checkpoint(format!("config parse: {e}")))?;
    let model = Rlt::new(config, device)?;
    model.varmap.load(path)?;
    Ok(model)
}
```

- [ ] **Step 4: Run tests** — expect `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add rust/rlt-core/src/checkpoint.rs rust/rlt-core/src/lib.rs rust/rlt-core/tests/checkpoint.rs
git commit -m "rust(rlt-core): safetensors + JSON config checkpoints"
```

---

### Task 13: rlt-cli binary

**Files:**
- Modify: `rust/rlt-cli/src/main.rs`
- Test: `rust/rlt-cli/tests/cli.rs`

- [ ] **Step 1: Write the failing CLI integration test**

```rust
use std::path::PathBuf;
use std::process::Command;

fn bin() -> &'static str { env!("CARGO_BIN_EXE_rlt") }

fn tmpdir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("rlt_cli_test_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn init_train_generate_roundtrip() {
    let dir = tmpdir();
    let model = dir.join("model.st");
    let corpus = dir.join("corpus.txt");
    std::fs::write(&corpus, "the quick brown fox jumps over the lazy dog. ".repeat(20)).unwrap();

    let init = Command::new(bin())
        .args(["init", "--out"])
        .arg(&model)
        .args(["--d-model", "64", "--layers", "2", "--heads", "4"])
        .output()
        .unwrap();
    assert!(init.status.success(), "init failed: {}", String::from_utf8_lossy(&init.stderr));

    let train = Command::new(bin())
        .args(["train", "--corpus"])
        .arg(&corpus)
        .args(["--init"])
        .arg(&model)
        .args(["--out"])
        .arg(&model)
        .args(["--steps", "3", "--seq-len", "48"])
        .output()
        .unwrap();
    assert!(train.status.success(), "train failed: {}", String::from_utf8_lossy(&train.stderr));

    let gen = Command::new(bin())
        .args(["generate", "--model"])
        .arg(&model)
        .args(["--prompt", "the", "--max-tokens", "8", "--temperature", "0"])
        .output()
        .unwrap();
    assert!(gen.status.success(), "generate failed: {}", String::from_utf8_lossy(&gen.stderr));
    let stdout = String::from_utf8_lossy(&gen.stdout);
    assert!(!stdout.trim().is_empty(), "generate produced no output");

    std::fs::remove_dir_all(&dir).ok();
}
```

- [ ] **Step 2: Run to verify failure** — CLI prints "not yet implemented", test fails.

- [ ] **Step 3: Implement `rust/rlt-cli/src/main.rs`**

```rust
//! rlt — CLI for the Recurrent Looped Transformer (rlt-core).

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rlt_core::{load_checkpoint, save_checkpoint, ByteTokenizer, Device, Rlt, RltConfig, Sampler};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Parser)]
#[command(name = "rlt", version, about = "Recurrent Looped Transformer (RLT) CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Write a randomly initialized checkpoint.
    Init {
        /// Output checkpoint path.
        #[arg(long)]
        out: PathBuf,
        /// Model width.
        #[arg(long, default_value_t = 256)]
        d_model: usize,
        /// Encoder = decoder layer count.
        #[arg(long, default_value_t = 4)]
        layers: usize,
        /// Attention heads.
        #[arg(long, default_value_t = 8)]
        heads: usize,
        /// SWA window size.
        #[arg(long, default_value_t = 64)]
        window: usize,
        /// Use untied decoder weights (default: tied, paper §2.6).
        #[arg(long)]
        untied: bool,
    },
    /// Train on a text corpus (byte-level) with full BPTT.
    Train {
        /// Corpus text file.
        #[arg(long)]
        corpus: PathBuf,
        /// Checkpoint to start from (omit to train a fresh default model).
        #[arg(long)]
        init: Option<PathBuf>,
        /// Output checkpoint path.
        #[arg(long)]
        out: PathBuf,
        /// Optimizer steps.
        #[arg(long, default_value_t = 100)]
        steps: usize,
        /// Learning rate.
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        /// Sequence window length (independent segments, paper §5.1).
        #[arg(long, default_value_t = 128)]
        seq_len: usize,
        /// Windows per optimizer step (loss-averaged).
        #[arg(long, default_value_t = 1)]
        batch: usize,
        /// Print loss every N steps.
        #[arg(long, default_value_t = 10)]
        log_every: usize,
    },
    /// Generate text from a prompt.
    Generate {
        /// Checkpoint path.
        #[arg(long)]
        model: PathBuf,
        /// Prompt text.
        #[arg(long)]
        prompt: String,
        /// Maximum tokens to generate.
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        /// Sampling temperature (0 = greedy).
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        /// Optional top-k.
        #[arg(long)]
        top_k: Option<usize>,
        /// RNG seed.
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Init { out, d_model, layers, heads, window, untied } => {
            let cfg = RltConfig {
                d_model,
                n_heads: heads,
                n_encoder_layers: layers,
                n_decoder_layers: layers,
                window,
                tied: !untied,
                ..Default::default()
            };
            cfg.validate().context("invalid config")?;
            let model = Rlt::new(cfg, Device::Cpu)?;
            save_checkpoint(&model, &out)?;
            println!("wrote {}", out.display());
        }
        Commands::Train { corpus, init, out, steps, lr, seq_len, batch, log_every } => {
            let text = std::fs::read_to_string(&corpus)
                .with_context(|| format!("reading {}", corpus.display()))?;
            let tok = ByteTokenizer::new(rlt_core::MIN_VOCAB)?;
            let tokens = tok.encode(&text, true, true);
            let model = match init {
                Some(p) => load_checkpoint(&p, Device::Cpu)?,
                None => Rlt::new(RltConfig::default(), Device::Cpu)?,
            };
            let mut opt = candle_nn::AdamW::new(
                model.varmap.all_vars(),
                candle_nn::ParamsAdamW { lr, ..Default::default() },
            )?;
            let windows: Vec<Vec<u32>> = tokens
                .chunks(seq_len)
                .filter(|c| c.len() >= 2)
                .map(|c| c.to_vec())
                .collect();
            anyhow::ensure!(!windows.is_empty(), "corpus too short for seq-len {seq_len}");
            let mut step = 0;
            while step < steps {
                let group: Vec<&Vec<u32>> =
                    (0..batch).map(|b| &windows[(step + b) % windows.len()]).collect();
                let loss = if group.len() == 1 {
                    model.forward_loss(group[0], None)?
                } else {
                    let mut total = model.forward_loss(group[0], None)?;
                    for w in &group[1..] {
                        total = (total + model.forward_loss(w, None)?)?;
                    }
                    total.affine(1.0 / group.len() as f64, 0.0)?
                };
                let val = loss.to_scalar::<f32>()? as f64;
                opt.backward_step(&loss)?;
                step += 1;
                if step % log_every == 0 || step == 1 || step == steps {
                    println!("step {step}/{steps} loss {val:.4}");
                }
            }
            save_checkpoint(&model, &out)?;
            println!("wrote {}", out.display());
        }
        Commands::Generate { model, prompt, max_tokens, temperature, top_k, seed } => {
            let model = load_checkpoint(&model, Device::Cpu)?;
            let tok = ByteTokenizer::new(rlt_core::MIN_VOCAB)?;
            let prompt_tokens = tok.encode(&prompt, true, false);
            let sampler = if temperature <= 0.0 {
                Sampler::Greedy
            } else {
                Sampler::Sample { temperature: temperature as f32, top_k }
            };
            let mut rng = StdRng::seed_from_u64(seed);
            let (sampled, _) = model.generate(&prompt_tokens, max_tokens, &sampler, &mut rng, true)?;
            let ids: Vec<u32> = sampled.iter().map(|t| t.token).collect();
            let logprob_sum: f64 = sampled.iter().map(|t| t.logprob as f64).sum();
            println!("{}", tok.decode(&ids));
            eprintln!("tokens: {}  logprob sum: {logprob_sum:.3}", ids.len());
        }
    }
    Ok(())
}
```

- [ ] **Step 4: Run the CLI test** — `cd rust && cargo test -p rlt-cli 2>&1 | tail -8`
Expected: `1 passed`.

- [ ] **Step 5: Manual smoke** (from `rust/`):

```bash
cargo run -q -p rlt-cli -- init --out /tmp/rlt_demo.st --d-model 128 --layers 2 --heads 4
printf 'to be or not to be, that is the question. ' > /tmp/rlt_corpus.txt
cargo run -q -p rlt-cli -- train --corpus /tmp/rlt_corpus.txt --init /tmp/rlt_demo.st --out /tmp/rlt_demo.st --steps 5 --seq-len 32 --log-every 1
cargo run -q -p rlt-cli -- generate --model /tmp/rlt_demo.st --prompt "to be" --max-tokens 16 --temperature 0
```

Expected: init prints "wrote ...", train prints 5 loss lines, generate prints text
(may be gibberish — random init, 5 steps).

- [ ] **Step 6: Commit**

```bash
git add rust/rlt-cli/src/main.rs rust/rlt-cli/tests/cli.rs
git commit -m "rust(rlt-cli): init/train/generate CLI"
```

---

### Task 14: READMEs and crate docs

**Files:**
- Create: `rust/rlt-core/README.md`, `rust/rlt-cli/README.md`

- [ ] **Step 1: Write `rust/rlt-core/README.md`**

````markdown
# rlt-core

Recurrent Looped Transformer (RLT) in pure Rust ([candle](https://github.com/huggingface/candle)):
a causal encoder builds global key–value memory; a recurrent decoder merges each
token's encoder representation with the previous decoder output and carries the
complete state `H_t = (s_t, C_t^D)` (recurrent output + per-layer sliding-window
KV caches) across every prompt and response token.

Reference: [*Recurrent Looped Transformer*](https://github.com/yifanzhang-pro/recurrent-looped-tranformer) (Zhang, 2026).

## Features

- Paper-faithful execution: parallel causal encoder prefill, recurrent decoder over
  every token, incremental one-token stepping, no prompt/response state reset.
- Gated merge feedback (paper Eq. 2.9–2.11) with learned initial state s⋆.
- Sliding-window decoder self-attention + encoder-memory cross-attention.
- Tied configuration (paper §2.6): decoder SWA/FFN reuse compatible encoder weights.
- Full-BPTT training with optional SFT-style loss masking (nothing detached).
- Current-policy RL replay: `replay()` rebuilds the full history under current
  parameters and returns log-probs + importance ratios for action tokens.
- Byte-level tokenizer, greedy/temperature/top-k sampling, safetensors checkpoints.
- CPU by default; `cuda` feature for NVIDIA GPUs.

## Quickstart

```rust,no_run
use rlt_core::{Rlt, RltConfig, ByteTokenizer, Device, Sampler};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() -> rlt_core::Result<()> {
    let model = Rlt::new(RltConfig::default(), Device::Cpu)?;
    let tok = ByteTokenizer::new(258)?;
    let prompt = tok.encode("The recurrent state", true, false);
    let mut rng = StdRng::seed_from_u64(0);
    let (tokens, _state) = model.generate(&prompt, 32, &Sampler::Greedy, &mut rng, true)?;
    println!("{}", tok.decode(&tokens.iter().map(|t| t.token).collect::<Vec<_>>()));
    Ok(())
}
```

## Training

```rust,no_run
# use rlt_core::{Rlt, RltConfig, Device};
# use candle_nn::{AdamW, Optimizer};
# let model = Rlt::new(RltConfig::default(), Device::Cpu)?;
let mut opt = AdamW::new(1e-3, model.varmap.all_vars())?;
let loss = model.train_step(&[1, 2, 3, 4, 5], None, &mut opt)?;
# Ok::<(), rlt_core::RltError>(())
```

`forward_loss(tokens, Some(&mask))` applies an SFT-style mask (1.0 = supervise);
masked tokens still update state (paper §5.2).

## License

MIT OR Apache-2.0.
````

- [ ] **Step 2: Write `rust/rlt-cli/README.md`**

````markdown
# rlt-cli

CLI for the [Recurrent Looped Transformer](https://github.com/yifanzhang-pro/recurrent-looped-tranformer)
(`rlt-core`).

## Install

```bash
cargo install rlt-cli
```

## Usage

```bash
# random checkpoint (defaults: --d-model 256 --layers 4 --heads 8 --window 64)
rlt init --out model.st

# train on a corpus (byte-level, full BPTT, independent seq-len segments)
rlt train --corpus corpus.txt --init model.st --out model.st --steps 1000 --lr 1e-3 --seq-len 128

# generate (temperature 0 = greedy)
rlt generate --model model.st --prompt "To be" --max-tokens 64 --temperature 0.8 --seed 0
```

## License

MIT OR Apache-2.0.
````

- [ ] **Step 3: Verify docs build** — `cd rust && cargo doc --no-deps -p rlt-core 2>&1 | tail -3`; expect `Finished`. Fix any rustdoc warnings (broken intra-doc links etc.).

- [ ] **Step 4: Commit**

```bash
git add rust/rlt-core/README.md rust/rlt-cli/README.md
git commit -m "rust: crate READMEs"
```

---

### Task 15: Pre-publish verification

- [ ] **Step 1: Full test suite**

Run: `cd rust && cargo test --workspace 2>&1 | tail -25`
Expected: all suites `ok` — api_probe (2), config (3), tokenizer (3), nn (5),
encoder (1), execution (4), gradients (5), sampling (4), replay (2), checkpoint (2),
cli (1). Failures must be fixed, never skipped.

- [ ] **Step 2: Clippy**

Run: `cd rust && cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -5`
Expected: exit 0. Fix all lints (needless borrows, redundant clones, etc.).

- [ ] **Step 3: Format**

Run: `cd rust && cargo fmt --all && cargo fmt --all -- --check`
Expected: exit 0.

```bash
git add -A rust/
git commit -m "rust: pre-publish checks (clippy, fmt)" || true
```

- [ ] **Step 4: Package dry runs**

```bash
cd rust/rlt-core && cargo package --allow-dirty --no-verify 2>&1 | tail -3
cd ../rlt-cli && cargo package --allow-dirty --no-verify 2>&1 | tail -3
```
Expected: `Packaged N files` for both, no errors.

- [ ] **Step 5: Verify license files packaged**

Run (in each crate dir): `cargo package --list --allow-dirty 2>/dev/null | grep LICENSE`
Expected: `LICENSE-APACHE` and `LICENSE-MIT` listed for both crates.

---

### Task 16: Publish

**Context:** crates.io releases are permanent (yank-only). Publishing is explicitly
user-approved. Token exists in `~/.cargo/credentials.toml`.

- [ ] **Step 1: Commit any remaining changes**

```bash
git status --short
git add -A && git commit -m "rust: release 0.1.0" || true
```

- [ ] **Step 2: Publish the library first**

```bash
cd rust/rlt-core && cargo publish 2>&1 | tail -5
```
Success ends with `Uploading rlt-core v0.1.0 to crates.io` / exit 0. If metadata or
size errors appear, fix and re-run.

Index check after ~30s:
`curl -s -H "User-Agent: rlt-check (contact: local)" https://crates.io/api/v1/crates/rlt-core | head -c 150` — expect crate JSON.

- [ ] **Step 3: Publish the CLI**

```bash
cd rust/rlt-cli && cargo publish 2>&1 | tail -5
```
Same expectation for `rlt-cli`.

- [ ] **Step 4: Verify docs.rs builds**

`curl -s "https://docs.rs/crate/rlt-core/0.1.0/builds" | grep -o '"build_status":"[^"]*"' | head -2` — expect `"Success"` (may take ~5 min; poll twice).
Same for `rlt-cli`.

- [ ] **Step 5: Registry install smoke test**

```bash
cargo install rlt-cli --root /tmp/rlt-install --force 2>&1 | tail -3
/tmp/rlt-install/bin/rlt --version
```
Expected: version prints (`rlt-cli 0.1.0`).

- [ ] **Step 6: Tag and push**

```bash
git tag -a v0.1.0 -m "rlt-core + rlt-cli 0.1.0 (crates.io release)"
git push origin master
git push origin v0.1.0
```
Expected: pushes succeed (pre-approved). If rejected, report and keep the tag local.

- [ ] **Step 7: Final report** — summarize: crate URLs, docs.rs URLs, test counts,
install command, spec §10 limitations restated.
