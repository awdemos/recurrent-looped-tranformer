//! Execution paths (paper §2.3–2.5, App. A): prefill, incremental step, training.

use candle_core::{D, Tensor};
use candle_nn::{AdamW, Module, Optimizer};

use crate::model::Rlt;
use crate::state::{GroupKv, RltState};
use crate::{Result, RltError};

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
                .collect::<candle_core::Result<Vec<_>>>()?,
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
        Ok(num.broadcast_div(&den)?)
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
}
