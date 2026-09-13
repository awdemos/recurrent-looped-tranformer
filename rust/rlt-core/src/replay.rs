//! Current-policy RL replay (paper §5.3, §A.4).
//!
//! The sampler records tokens, an action mask, and behavior log-probabilities under
//! the actual sampling distribution µ (including any temperature/top-k transforms).
//! The trainer replays the complete history under current parameters Θ — full
//! prompt recurrence, no stale caches, no detach — and forms token ratios
//! `r_i = exp(log p_Θ − log µ)` for policy actions only.

use candle_core::{D, Tensor};

use crate::model::Rlt;
use crate::{Result, RltError};

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
    /// For stochastic sampling these MUST describe the actual sampling
    /// distribution (temperature/top-k transforms included) — record
    /// `Sampler::behavior_logprob(&logits, token)` per step, not
    /// `SampledToken.logprob`, unless sampling untruncated at temperature 1.0
    /// (see `crate::Sampler`).
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
    /// Ratios are detached constants; build a differentiable surrogate from
    /// `current_logprobs` and `Rollout.behavior_logprobs`.
    pub ratios: Vec<f32>,
}

/// Replay a rollout under the model's current parameters (paper §A.4).
///
/// Every token — prompt and response, action or external — is consumed exactly
/// once; actions are scored from the preceding state before being consumed.
/// Gradients flow through the full prompt+response replay (full BPTT, paper
/// §A.4); nothing is detached.
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
            let scalar = lp.squeeze(0)?; // ()
            let cur = scalar.to_scalar::<f32>()?;
            ratios.push((cur - rollout.behavior_logprobs[i]).exp());
            cur_logp.push(scalar);
        }
        model.step(tok, &mut state)?;
    }
    if cur_logp.is_empty() {
        return Err(RltError::Replay("rollout has no action tokens".into()));
    }
    Ok(ReplayResult { current_logprobs: Tensor::stack(&cur_logp, 0)?, ratios })
}
