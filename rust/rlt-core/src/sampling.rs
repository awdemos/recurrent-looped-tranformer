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
        /// Temperature (finite, > 0); divides logits before softmax.
        temperature: f32,
        /// Optional top-k truncation.
        top_k: Option<usize>,
    },
}

impl Sampler {
    /// log-probability of `token` under this sampler's ACTUAL sampling distribution
    /// (paper §5.3: behavior log-probabilities must describe the distribution that
    /// produced the action, including temperature and truncation/renormalization).
    ///
    /// - `Greedy`: point mass at the argmax — 0.0 for the argmax, −∞ otherwise.
    /// - `Sample`: logits scaled by 1/temperature, optionally top-k truncated; a
    ///   token outside the candidate set has probability 0 (−∞ log-prob). For
    ///   temperature 1.0 without top-k this equals the model's full-vocab log-prob.
    pub fn behavior_logprob(&self, logits: &Tensor, token: u32) -> Result<f32> {
        let v = logits.squeeze(0)?.to_vec1::<f32>()?;
        match self {
            Sampler::Greedy => {
                let (idx, _) = argmax(&v)?;
                Ok(if idx as u32 == token { 0.0 } else { f32::NEG_INFINITY })
            }
            Sampler::Sample { temperature, top_k } => {
                check_temperature(*temperature)?;
                let cand = candidates(&v, *temperature, *top_k);
                if cand.is_empty() {
                    return Err(RltError::State("empty logits".into()));
                }
                let Some(j) = cand.iter().position(|&(t, _)| t == token) else {
                    return Ok(f32::NEG_INFINITY);
                };
                let logp = log_softmax_vec(&cand.iter().map(|&(_, x)| x).collect::<Vec<_>>());
                Ok(logp[j])
            }
        }
    }
}

/// A sampled token and its log-probability under the model's full-vocab
/// distribution.
#[derive(Clone, Debug)]
pub struct SampledToken {
    /// Sampled token id.
    pub token: u32,
    /// The model's full-vocab log-probability of this token. Valid as the RL
    /// behavior log-probability ONLY for untruncated sampling at temperature 1.0;
    /// under any other sampler use [`Sampler::behavior_logprob`], which describes
    /// the actual sampling distribution (paper §5.3).
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
            let (idx, _) = argmax(&v)?;
            Ok(SampledToken { token: idx as u32, logprob: logp_full[idx] })
        }
        Sampler::Sample { temperature, top_k } => {
            check_temperature(*temperature)?;
            let cand = candidates(&v, *temperature, *top_k);
            if cand.is_empty() {
                return Err(RltError::State("empty logits".into()));
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

fn check_temperature(temperature: f32) -> Result<()> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(RltError::Config("sampling temperature must be finite and > 0".into()));
    }
    Ok(())
}

fn argmax(v: &[f32]) -> Result<(usize, f32)> {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, &x)| (i, x))
        .ok_or_else(|| RltError::State("empty logits".into()))
}

/// Candidate list for `Sample`: logits scaled by 1/temperature, optionally
/// truncated to the top-k highest (k clamped to >= 1). Renormalization is the
/// caller's job (`softmax_vec` / `log_softmax_vec`).
fn candidates(v: &[f32], temperature: f32, top_k: Option<usize>) -> Vec<(u32, f32)> {
    let mut cand: Vec<(u32, f32)> = v
        .iter()
        .enumerate()
        .map(|(i, &x)| (i as u32, x / temperature))
        .collect();
    if let Some(k) = top_k {
        cand.sort_by(|a, b| b.1.total_cmp(&a.1));
        cand.truncate(k.max(1));
    }
    cand
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
