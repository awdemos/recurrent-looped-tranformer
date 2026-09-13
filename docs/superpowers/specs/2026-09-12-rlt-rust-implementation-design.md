# RLT Rust Implementation — Design Spec

**Date:** 2026-09-12
**Status:** Approved (user sign-off on structure, scope, backend, CLI, license)
**Source spec:** `Recurrent_Looped_Transformer.pdf` (Zhang, 2026) — equations/sections cited below.

## 1. Goal

A Rust implementation of the Recurrent Looped Transformer (RLT) as published library
crate(s) on crates.io, plus a CLI binary built on the library. Covers the full paper
scope: forward execution, full-BPTT training, and current-policy RL replay APIs.

## 2. Decisions (locked)

| Question | Decision |
|---|---|
| Scope | Full: forward inference + training (full BPTT) + RL replay APIs |
| Backend | candle (candle-core, candle-nn) — pure Rust autograd, CPU default, optional CUDA |
| "Publish" | Real `cargo publish` to crates.io (token present in `~/.cargo/credentials.toml`) |
| Structure | Cargo workspace: `rlt-core` (library) + `rlt-cli` (binary), both published |
| Binary | CLI with `train` / `generate` / `init` subcommands |
| License | MIT OR Apache-2.0 (dual), at `rust/LICENSE-MIT` + `rust/LICENSE-APACHE` |
| Crate names | `rlt-core`, `rlt-cli` (both verified free on crates.io 2026-09-12; `rlt` is taken) |
| Dropout | Excluded from v1 — deterministic forward (paper §A.4 avoids ambiguity) |

## 3. Workspace layout

This repo (a paper/GitHub-Pages repo) gains a `rust/` subdirectory; the site at repo
root is untouched.

```
rust/
├── Cargo.toml            # virtual workspace; [workspace.package] shares version/license/edition
├── LICENSE-MIT
├── LICENSE-APACHE
├── rlt-core/             # library crate (published first)
│   ├── Cargo.toml
│   ├── README.md
│   ├── src/lib.rs        # public API re-exports
│   ├── src/config.rs     # RltConfig
│   ├── src/model.rs      # Rlt: encoder stack, decoder stack, merge, readout
│   ├── src/attention.rs  # RoPE, causal / SWA / cross attention
│   ├── src/state.rs      # RltState: complete decoder state H_t + encoder cache + memory
│   ├── src/execution.rs  # prefill, step, training unroll, sampling
│   ├── src/replay.rs     # Rollout, ReplayResult, importance ratios
│   ├── src/tokenizer.rs  # byte-level tokenizer with BOS/EOS
│   ├── src/checkpoint.rs # safetensors save/load
│   └── tests/*.rs        # integration tests (§8)
└── rlt-cli/              # binary crate (published second; `cargo install rlt-cli`)
    ├── Cargo.toml
    ├── README.md
    └── src/main.rs       # clap CLI; depends on rlt-core via { version, path }
```

Rationale for the split (production-scaling goal): CLI-only dependencies (clap,
tracing-subscriber, indicatif) compile only for `rlt-cli` consumers; downstream
production users of `rlt-core` compile only the lean library. The CLI is a genuine
external consumer (version + path dep), enforcing public-API discipline. Future
production components (`rlt-server`, data pipelines, FFI) attach as sibling crates.

## 4. rlt-core architecture (paper mapping)

### 4.1 Config

```rust
pub struct RltConfig {
    pub d_model: usize,          // residual width d
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_encoder_layers: usize, // L_E
    pub n_decoder_layers: usize, // L_D
    pub window: usize,           // W, includes current token
    pub memory_groups: usize,    // G (§2.2); G=1 shares memory across decoder layers
    pub vocab_size: usize,       // byte-level: 256 + special tokens
    pub tied: bool,              // §2.6 tied configuration
    pub feedback_alpha: f64,     // α, merge feedback scale
    pub max_seq_len: usize,      // RoPE table bound
}
```

Demo-scale defaults: `d_model=256, n_heads=8, d_ff=1024, L_E=L_D=4, W=64, G=1,
tied=true, α=0.1`. The paper's 48+48 is reachable via config but not demo-default.

### 4.2 Encoder (§2.2)

Embedding → L_E pre-norm blocks: causal MHA (RoPE on q/k) → RMSNorm → residual →
pre-norm FFN (plain GELU MLP: Linear→GELU→Linear) → residual. Produces `e_t` for
every position; runs token-parallel during prefill.
Incremental mode uses a per-layer causal KV cache `C^E` (Eq. 2.8).

### 4.3 Global KV memory (Eq. 2.2)

Per group g: `k_t^g = W_K^g · RoPE(RMSNorm_E(e_t), t)`,
`v_t^g = W_V^g · RMSNorm_E(e_t)`; `M_{≤t} = {(k_j^g, v_j^g)}_{j=1..t}`.
Decoder layer ℓ reads group `g(ℓ) = ℓ % G`. Prefix-restricted: layer at position t
attends only to memory entries `j ≤ t`.

### 4.4 Merge (Eq. 2.9–2.11)

`r_{t-1} = RMSNorm_s(s_{t-1})`
`g_t = σ(W_g · [e_t ; r_{t-1}] + b_g)`   (W_g ∈ R^{d×2d})
`u_t = e_t + α · g_t ⊙ (W_s · r_{t-1})` (W_s ∈ R^{d×d})

### 4.5 Decoder block (Eq. 2.12–2.16), fixed sublayer order

Input `z^{ℓ-1}`:
1. **Causal SWA self-attention**: `q = W_q RoPE(RMSNorm_{S,ℓ}(z), t)`,
   `k = W_k RoPE(RMSNorm_{S,ℓ}(z), t)`, `v = W_v RMSNorm_{S,ℓ}(z)`; attend to
   `{k_j, v_j}_{j=max(1, t−W+1)..t}` (historical from `C^D_{t-1}`, current formed
   from layer input — no circular dependency); residual.
2. **Cross-attention** to `M_{≤t}` with per-layer query/output projections;
   residual.
3. **FFN** on `RMSNorm_{D,ℓ}(·)`; residual.

`s_t = z^{L_D}`. After each update, layer cache retains positions
`max(1, t−W+2)..t` (at most W−1 entries) — the paper's eviction convention.

### 4.6 Readout (Eq. 2.6)

`p(x_{t+1}|x_{1:t}) = softmax(W_o · RMSNorm_o(s_t))`.

### 4.7 State

```rust
pub struct RltState {
    pub recurrent_output: Tensor,      // s_t
    pub decoder_swa: Vec<LayerKv>,     // C_t^D, per decoder layer, ≤ W−1 entries
    pub encoder_cache: Vec<LayerKv>,   // C_t^E, per encoder layer (full prefix)
    pub memory: Vec<GroupKv>,          // M_{≤t}, per memory group
    pub position: usize,               // t
}
// H_0 = (s_star, ∅); s_star is a learned parameter.
```

### 4.8 Tied configuration (§2.6)

`tied=true`: decoder SWA layer ℓ reuses encoder layer ℓ's attention projections
(W_q, W_k, W_v, W_o) and FFN weights — parameter reuse with different attention
wiring, NOT activation copying. Decoder cross-attention projections, all stage
norms, merge, and readout remain separate per layer. `tied=false`: fully untied
stacks. Memory-group sharing (G) is an independent axis and never merges the
layerwise SWA caches.

## 5. Execution paths

- `Rlt::prefill(&self, tokens) -> Result<(Vec<Tensor> /*logits per pos*/, RltState)>`
  Parallel causal encoder, then the decoder transition for **every** prompt token in
  order (full prompt recurrence, §A.1). No ordinary parallel-SWA shortcut: historical
  decoder KV comes only from preceding recurrent updates.
- `Rlt::step(&self, token, state: &mut RltState) -> Result<Tensor>`
  Incremental: encoder step on `C^E` → memory append → merge → decoder blocks with SWA
  cache → readout. Each consumed token gets exactly one recurrent update and one KV
  insertion per SWA layer (§A.2).
- `Rlt::train_step(&self, tokens, loss_mask: Option<&[f32]>, opt) -> Result<(Tensor, f64)>`
  Same transition unrolled with gradient tracking — full BPTT through `s_t`, decoder
  SWA KV, and encoder memory; nothing detached (§5.1–5.2, §5.4). `loss_mask` gives
  SFT-style assistant-only targets; masked tokens still update state.
  Loss: masked-mean NLL over next-token targets.
- Sampling: greedy / temperature / top-k, with optional log-prob capture (behavior
  recording for replay). RNG injectable for testability.

## 6. RL replay API (§5.3, §A.4)

```rust
pub struct Rollout {
    pub prompt_tokens: Vec<u32>,
    pub response_tokens: Vec<u32>,
    pub action_mask: Vec<bool>,        // true = policy action (assistant token)
    pub behavior_logprobs: Vec<f32>,   // log µ(y_i | ·), incl. sampling transforms
    pub metadata: SamplingMetadata,    // temperature/top-k/seed, for the record
}
pub struct ReplayResult {
    pub current_logprobs: Vec<f32>,    // differentiable graph under current Θ
    pub ratios: Vec<f32>,              // exp(log p_Θ − log µ) for action tokens only
}
pub fn replay(model: &Rlt, rollout: &Rollout) -> Result<ReplayResult>;
```

Rewards, advantages, clipping, KL: caller's responsibility (paper: no particular
reward function/clipping/KL is required by the architecture). Invariant: replay at
identical parameters and conventions ⇒ all ratios = 1 (tested).

## 7. CLI (rlt-cli)

Byte-level tokenizer (256 vocab + BOS/EOS) keeps everything self-contained.

- `rlt init --out model.st [--d-model N] [--layers N] [--window N] [--untied] [--seed S]`
  Write a random-init checkpoint.
- `rlt train --corpus FILE [--out model.st] [--init model.st] [--steps N] [--lr F]
  [--seq-len N] [--batch N]`
  Full-BPTT training on the corpus; periodic checkpointing; loss printed per N steps.
- `rlt generate --model model.st --prompt "..." [--max-tokens N] [--temperature F]
  [--top-k N] [--seed S]`
  Prefill + incremental decode, single-shot per invocation. (Multi-turn state
  continuity is exercised through the library API — `prefill`/`step` — not the CLI.)

## 8. Verification plan (all must pass before publish)

1. **Prefill ≡ incremental** (Prop. 3.1): identical tokens → logits match ≤ 1e-4 (fp32).
2. **Causality** (Prop. B.1): mutating token at position j changes no logit < j.
3. **SWA cache semantics** (§2.3–2.5): after processing n tokens, each decoder layer
   cache holds exactly `min(W−1, n)` historical positions (W=1 ⇒ permanently empty),
   and prefill≡incremental equivalence holds at small W (covers eviction correctness).
4. **Gradient check**: central finite differences vs. candle backward agree ≤ 1e-3
   on merge params (W_g, b_g, W_s, α-adjacent) and one attention projection.
5. **Masking**: SFT mask alters loss; logits/state of later positions unchanged
   relative to unmasked run.
6. **Replay**: same-params replay ⇒ ratios all 1.0; action tokens only.
7. **Checkpoint roundtrip**: save→load→identical logits.
8. **Train smoke**: toy corpus loss decreases monotonically over ~50 steps.
9. `cargo package` clean, docs build, `clippy -D warnings`, `fmt --check`.

## 9. Publishing plan

1. Fill metadata: description, license-file(s), homepage/repository = this repo,
   categories = `science`, `algorithms`; version via `[workspace.package]`.
2. Publish `rlt-core` first, then `rlt-cli` (path+version dep resolves to registry).
3. Post-publish: verify docs.rs build, `cargo install rlt-cli` smoke.
4. Known irreversibility: crates.io releases cannot be deleted, only yanked.

## 10. Risks / limitations (stated honestly, per the paper's own disclaimers)

- The recurrent decoder unroll is sequential; no wall-clock speedup is claimed
  (paper §3.2, §4: opportunities, not measured results).
- candle backward coverage: all ops used (matmul, softmax, RoPE via mul/slice/cat,
  RMSNorm) support backprop; gradient test (§8.4) guards this.
- CPU is the default target; the CUDA feature is provided but GPU correctness is
  validated opportunistically (this machine has an NVIDIA GPU), not guaranteed in CI.
- Analytical depth ≠ reasoning quality (paper §3.3); the crate implements the
  computation, it does not claim the paper's research goals are realized.

## 11. Out of scope (v1)

Dropout, GQA, MoE, quantized dtypes, distributed/multi-GPU training, HTTP serving,
Python bindings, HF-hub tokenizer integration, streaming server. The workspace
structure anticipates these as future sibling crates.
