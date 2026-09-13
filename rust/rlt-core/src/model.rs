//! The RLT model: encoder stack, decoder stack, merge, readout.

use candle_core::{D, Device, DType, Tensor};
use candle_nn::{Embedding, Init, Linear, Module, VarBuilder, VarMap};

use crate::config::RltConfig;
use crate::nn::{self, RotaryEmbedding, NORM_EPS};
use crate::state::{GroupKv, LayerKv};
use crate::{Result, RltError};

/// Bias-free linear with uniform init. Weight layout follows candle's
/// convention: (out_dim, in_dim), applied as `y = x @ w.t()`.
fn linear(vb: VarBuilder, in_dim: usize, out_dim: usize) -> Result<Linear> {
    let weight = vb.get_with_hints(
        (out_dim, in_dim),
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
        Ok((x + ff)?)
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
        Ok((x + ff)?)
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
    pub(crate) mem_k: Vec<Linear>,
    pub(crate) mem_v: Vec<Linear>,
    merge_w_g: Linear,
    merge_b_g: Tensor,
    merge_w_s: Linear,
    readout: Linear,
    pub(crate) s_star: Tensor,
    pub(crate) rotary: RotaryEmbedding,
}

impl Rlt {
    /// Create a model with fresh (randomly initialized) parameters.
    pub fn new(config: RltConfig, device: Device) -> Result<Self> {
        config.validate()?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let d = config.d_model;
        let emb = Embedding::new(
            vb.pp("embedding").get_with_hints(
                (config.vocab_size, d),
                "embeddings",
                // candle 0.11 `Init::default()` is `Const(0)`: an implicit `get`
                // here would leave the model input-independent.
                Init::Uniform { lo: -0.02, up: 0.02 },
            )?,
            d,
        );
        let mut encoder = Vec::with_capacity(config.n_encoder_layers);
        for l in 0..config.n_encoder_layers {
            encoder.push(EncoderBlock::new(&config, vb.pp(format!("encoder.{l}")))?);
        }
        let mut decoder = Vec::with_capacity(config.n_decoder_layers);
        for l in 0..config.n_decoder_layers {
            decoder.push(DecoderBlock::new(&config, l, &encoder, vb.pp(format!("decoder.{l}")))?);
        }
        let mut mem_k = Vec::with_capacity(config.memory_groups);
        let mut mem_v = Vec::with_capacity(config.memory_groups);
        for g in 0..config.memory_groups {
            mem_k.push(linear(vb.pp(format!("mem.{g}.k")), d, d)?);
            mem_v.push(linear(vb.pp(format!("mem.{g}.v")), d, d)?);
        }
        let merge_w_g = linear(vb.pp("merge.w_g"), 2 * d, d)?;
        let merge_b_g = vb.pp("merge").get_with_hints((d,), "b_g", Init::Const(0.))?;
        let merge_w_s = linear(vb.pp("merge.w_s"), d, d)?;
        let readout = linear(vb.pp("readout"), d, config.vocab_size)?;
        let s_star = vb.pp("state").get_with_hints((d,), "s_star", Init::Const(0.))?;
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
                &self.mem_k[g].forward(&normed)?,
                self.config.n_heads,
            )?;
            let v = nn::split_heads(
                &self.mem_v[g].forward(&normed)?,
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
        Ok((e_t + scaled)?)
    }

    /// Readout logits (paper Eq. 2.6). `s_t`: (1,D) → (1, vocab).
    pub fn logits(&self, s_t: &Tensor) -> Result<Tensor> {
        Ok(self.readout.forward(&nn::rms_norm(s_t, NORM_EPS)?)?)
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
        Ok(z.squeeze(0)?)
    }
}

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
        // W=1 keeps no history: the cache stays permanently empty, represented as None.
        let keep = self.window.saturating_sub(1);
        *cache = if keep == 0 {
            None
        } else {
            Some(LayerKv { k: k_all, v: v_all }.trimmed(keep)?)
        };
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
        Ok((z + ff)?)
    }
}
