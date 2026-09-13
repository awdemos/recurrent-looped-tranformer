use candle_core::Device;
use rlt_core::{Rlt, RltConfig};

fn tiny_config() -> RltConfig {
    RltConfig {
        d_model: 32,
        n_heads: 2,
        d_ff: 64,
        n_encoder_layers: 2,
        n_decoder_layers: 2,
        window: 3,
        memory_groups: 1,
        vocab_size: 258,
        tied: false,
        feedback_alpha: 0.1,
        max_seq_len: 64,
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

/// Paper Proposition B.1: position j logits depend only on tokens ≤ J.
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
    let model = Rlt::new(cfg.clone(), Device::Cpu).unwrap();
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
