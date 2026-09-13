//! Tensor-level primitives: RMSNorm, rotary embeddings, attention.

use candle_core::{bail, D, Device, Result, Tensor};

/// RMSNorm epsilon used throughout the model.
pub const NORM_EPS: f32 = 1e-5;

/// Unweighted RMSNorm over the last dimension.
pub fn rms_norm(x: &Tensor, eps: f32) -> Result<Tensor> {
    let ms = x.powf(2.)?.mean(D::Minus1)?.unsqueeze(D::Minus1)?;
    let denom = ms.broadcast_add(&Tensor::new(eps, x.device())?)?.sqrt()?;
    x.broadcast_div(&denom)
}

/// Rotates pairs of channels: `[-x_{d/2..}, x_{0..d/2}]` (standard RoPE pairing).
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?.affine(-1.0, 0.0)?;
    Tensor::cat(&[&x2, &x1], D::Minus1)
}

/// Cached rotary embedding table (RoPE, base 10000).
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    /// Build tables for `dim` channels (must be positive even) up to `max_seq_len`.
    pub fn new(dim: usize, max_seq_len: usize, device: &Device) -> Result<Self> {
        if dim == 0 || dim % 2 != 0 {
            bail!("rotary dim must be a positive even number, got {dim}");
        }
        let base = 10_000f32;
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| base.powf(-(i as f32) * 2.0 / dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (dim / 2,), device)?;
        let pos = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(candle_core::DType::F32)?;
        let freqs = pos.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?; // (S, dim/2)
        Ok(Self { cos: freqs.cos()?, sin: freqs.sin()? })
    }

    /// Apply rotation to a single tensor (B, H, S, D) at absolute `offset`.
    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let seq = x.dim(2)?;
        let cos = self.cos.narrow(0, offset, seq)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, offset, seq)?.unsqueeze(0)?.unsqueeze(0)?;
        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?; // (1,1,S,D)
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?;
        x.broadcast_mul(&cos)?
            .broadcast_add(&rotate_half(x)?.broadcast_mul(&sin)?)
    }

    /// Apply rotation to query and key (B, H, S, D) at absolute `offset`.
    pub fn apply_qk(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        Ok((self.apply(q, offset)?, self.apply(k, offset)?))
    }
}

/// Scaled dot-product attention.
///
/// `q`: (B, H, Sq, D); `k`, `v`: (B, H, Sk, D). `mask` is an additive broadcastable
/// tensor (e.g. (1,1,Sq,Sk) from [`causal_mask`]), or `None` when every key is valid.
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let d = q.dim(D::Minus1)? as f64;
    let mut scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
    scores = scores.affine(1.0 / d.sqrt(), 0.0)?;
    if let Some(m) = mask {
        scores = scores.broadcast_add(m)?;
    }
    candle_nn::ops::softmax(&scores, D::Minus1)?.matmul(v)
}

/// Additive causal mask (1, 1, T, T): 0 on/below the diagonal, −∞ above.
pub fn causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mut rows = Vec::with_capacity(t);
    for i in 0..t {
        let mut row = vec![0f32; t];
        for j in (i + 1)..t {
            row[j] = f32::NEG_INFINITY;
        }
        rows.push(row);
    }
    Tensor::new(rows, device)?.unsqueeze(0)?.unsqueeze(0)
}

/// (B, S, D) -> (B, H, S, D/h).
pub fn split_heads(x: &Tensor, n_heads: usize) -> Result<Tensor> {
    let (b, s, d) = x.dims3()?;
    x.reshape((b, s, n_heads, d / n_heads))?.transpose(1, 2)
}

/// (B, H, S, D/h) -> (B, S, D).
pub fn merge_heads(x: &Tensor) -> Result<Tensor> {
    let (b, h, s, hd) = x.dims4()?;
    x.transpose(1, 2)?.reshape((b, s, h * hd))
}
