//! Sampling from readout logits: greedy, temperature, top-k.

use candle_core::Tensor;
use rand::rngs::StdRng;
use rand::RngExt;

use crate::{Result, RltError};

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
                cand.truncate((*k).max(1));
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
