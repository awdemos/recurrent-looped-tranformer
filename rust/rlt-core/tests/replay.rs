use candle_core::Device;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rlt_core::{replay, Rlt, RltConfig, Rollout, Sampler};

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
        sampling: rlt_core::SamplingMetadata {
            temperature: 0.0,
            top_k: None,
            seed: 7,
        },
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
        sampling: rlt_core::SamplingMetadata {
            temperature: 1.0,
            top_k: None,
            seed: 0,
        },
    };
    assert!(replay(&model, &rollout).is_err());
}

#[test]
fn replay_rejects_no_action_tokens() {
    let model = Rlt::new(tiny_config(), Device::Cpu).unwrap();
    let rollout = Rollout {
        prompt_tokens: vec![1, 2],
        response_tokens: vec![3, 4],
        action_mask: vec![false, false],
        behavior_logprobs: vec![0.0, 0.0],
        sampling: rlt_core::SamplingMetadata {
            temperature: 1.0,
            top_k: None,
            seed: 0,
        },
    };
    assert!(replay(&model, &rollout).is_err());
}

/// The replay result is fully differentiable: a surrogate-style loss through
/// current_logprobs reaches ALL parameters, including prompt-path embeddings
/// (paper §A.4: replay must differentiate through external-token updates too).
#[test]
fn replay_logprobs_support_full_backward() {
    let model = Rlt::new(tiny_config(), Device::Cpu).unwrap();
    let mut rng = StdRng::seed_from_u64(3);
    let prompt = vec![8u32, 7, 6];
    let (sampled, _) = model
        .generate(&prompt, 4, &Sampler::Greedy, &mut rng, false)
        .unwrap();
    let response: Vec<u32> = sampled.iter().map(|t| t.token).collect();
    let behavior: Vec<f32> = sampled.iter().map(|t| t.logprob).collect();
    let rollout = Rollout {
        prompt_tokens: prompt.clone(),
        response_tokens: response.clone(),
        action_mask: vec![true, false, true, true], // external token in the middle
        behavior_logprobs: behavior.clone(),
        sampling: rlt_core::SamplingMetadata {
            temperature: 0.0,
            top_k: None,
            seed: 3,
        },
    };
    let result = replay(&model, &rollout).unwrap();
    // surrogate: sum over action logprobs (importance ratios enter as detached
    // constants in the real surrogate, so a plain sum exercises the same graph)
    let loss = result.current_logprobs.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let vars = model.varmap.all_vars();
    let missing = vars
        .iter()
        .filter(|v| grads.get(v.as_tensor()).is_none())
        .count();
    assert_eq!(
        missing, 0,
        "some parameters received no gradient through replay"
    );
    // at least one gradient must be non-zero (a connected-but-zero graph would pass above)
    let any_nonzero = vars.iter().any(|v| {
        grads
            .get(v.as_tensor())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            > 0.0
    });
    assert!(any_nonzero, "all replay gradients are zero");
}
