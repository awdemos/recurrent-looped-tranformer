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
- CPU by default; `cuda` feature for NVIDIA GPUs (experimental, not validated in CI).

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
use rlt_core::{Rlt, RltConfig, Device};
use candle_nn::AdamW;

fn main() -> rlt_core::Result<()> {
    let model = Rlt::new(RltConfig::default(), Device::Cpu)?;
    let mut opt = AdamW::new(model.varmap.all_vars(), Default::default())?;
    let loss = model.train_step(&[1, 2, 3, 4, 5], None, &mut opt)?;
    Ok(())
}
```

`forward_loss(tokens, Some(&mask))` applies an SFT-style mask (1.0 = supervise);
masked tokens still update state (paper §5.2).

## RL replay

```rust,no_run
use rlt_core::{replay, Rollout};

fn example(model: &rlt_core::Rlt, rollout: Rollout) -> rlt_core::Result<()> {
    let result = replay(model, &rollout)?;
    // result.current_logprobs: (A,) graph-connected current-policy log-probs
    // result.ratios: detached importance ratios r_i = exp(log p_Θ − log µ)
    Ok(())
}
```

Behavior log-probabilities must describe the actual sampling distribution; for
stochastic sampling record `Sampler::behavior_logprob(&logits, token)` per step
(paper §5.3).

## License

MIT OR Apache-2.0.
