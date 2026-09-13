use candle_core::{D, Device, Tensor};
use candle_nn::ops::log_softmax;
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
    let lp = log_softmax(&logits, D::Minus1).unwrap().to_vec2::<f32>().unwrap()[0][1];
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
