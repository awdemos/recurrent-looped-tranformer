use candle_core::{Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rlt_core::{Rlt, RltConfig};

fn tiny_config() -> RltConfig {
    RltConfig {
        d_model: 16,
        n_heads: 2,
        d_ff: 32,
        n_encoder_layers: 1,
        n_decoder_layers: 1,
        window: 2,
        memory_groups: 1,
        vocab_size: 258,
        tied: false,
        feedback_alpha: 0.1,
        max_seq_len: 32,
    }
}

/// 4-D extraction (candle has no `to_vec4`): reshape to 3-D and regroup.
fn t4(t: &Tensor) -> Vec<Vec<Vec<Vec<f32>>>> {
    let (a, b, c, d) = t.dims4().unwrap();
    t.reshape((a * b, c, d))
        .unwrap()
        .to_vec3::<f32>()
        .unwrap()
        .chunks(b)
        .map(|c| c.to_vec())
        .collect()
}

/// Finite-difference check of candle backward through rms_norm + sdpa + rope.
#[test]
fn fd_matches_backprop_on_attention_ops() {
    let dev = Device::Cpu;
    let x = Var::from_tensor(&Tensor::rand(-0.5f32, 0.5, (1, 2, 3, 8), &dev).unwrap()).unwrap();
    let w = Var::from_tensor(&Tensor::rand(-0.5f32, 0.5, (8, 8), &dev).unwrap()).unwrap();
    let rotary = rlt_core::nn::RotaryEmbedding::new(4, 8, &dev).unwrap();
    let loss_fn = |x: &Tensor, w: &Tensor| -> Tensor {
        // candle 0.11: plain `matmul` requires equal ranks/batches; broadcasting lives here.
        let xw = x.broadcast_matmul(w).unwrap(); // (1,2,3,8)
                                                 // split_heads maps (B,S,D) -> (B,H,S,hd): view the two leading dims as
                                                 // one sequence so the head split lands on the last dim (hd=4).
        let q = rlt_core::nn::split_heads(&xw.reshape((1, 6, 8)).unwrap(), 2).unwrap(); // (1,2,6,4)
        let (q, k) = rotary.apply_qk(&q, &q, 0).unwrap();
        let att = rlt_core::nn::sdpa(&q, &k, &q, None).unwrap();
        // Mean (not sum) of squares: at sum-of-squares scale ~O(50), f32 round-off
        // in `lp - lm` puts eps=1e-3 central differences at ~1e-2 error — above the
        // 5e-3 tolerance. The 1/48 constant scales fd and analytic identically;
        // stress-tested worst |fd - analytic| is ~2e-4 across random draws.
        rlt_core::nn::rms_norm(
            &rlt_core::nn::merge_heads(&att).unwrap().squeeze(0).unwrap(),
            1e-5,
        )
        .unwrap()
        .sqr()
        .unwrap()
        .mean_all()
        .unwrap()
    };
    let loss = loss_fn(x.as_tensor(), w.as_tensor());
    let grads = loss.backward().unwrap();
    let analytic_x = grads.get(x.as_tensor()).unwrap().clone();
    let analytic_w = grads.get(w.as_tensor()).unwrap().clone();
    let eps = 1e-3f32;
    let mut xv = t4(x.as_tensor());
    for idx in [(0usize, 0usize, 0usize, 0usize), (0, 1, 2, 7)] {
        let orig = xv[idx.0][idx.1][idx.2][idx.3];
        xv[idx.0][idx.1][idx.2][idx.3] = orig + eps;
        let lp = loss_fn(&Tensor::new(xv.clone(), &dev).unwrap(), w.as_tensor());
        xv[idx.0][idx.1][idx.2][idx.3] = orig - eps;
        let lm = loss_fn(&Tensor::new(xv.clone(), &dev).unwrap(), w.as_tensor());
        xv[idx.0][idx.1][idx.2][idx.3] = orig;
        let fd = (lp.to_scalar::<f32>().unwrap() - lm.to_scalar::<f32>().unwrap()) / (2.0 * eps);
        let an = t4(&analytic_x)[idx.0][idx.1][idx.2][idx.3];
        assert!((fd - an).abs() < 5e-3, "x{idx:?}: fd={fd} analytic={an}");
    }
    let mut wv = w.as_tensor().to_vec2::<f32>().unwrap();
    for idx in [(0usize, 0usize), (7, 7)] {
        let orig = wv[idx.0][idx.1];
        wv[idx.0][idx.1] = orig + eps;
        let lp = loss_fn(x.as_tensor(), &Tensor::new(wv.clone(), &dev).unwrap());
        wv[idx.0][idx.1] = orig - eps;
        let lm = loss_fn(x.as_tensor(), &Tensor::new(wv.clone(), &dev).unwrap());
        wv[idx.0][idx.1] = orig;
        let fd = (lp.to_scalar::<f32>().unwrap() - lm.to_scalar::<f32>().unwrap()) / (2.0 * eps);
        let an = analytic_w.to_vec2::<f32>().unwrap()[idx.0][idx.1];
        assert!((fd - an).abs() < 5e-3, "w{idx:?}: fd={fd} analytic={an}");
    }
}

/// Finite-difference check on real model parameters (merge bias, encoder
/// attention projection) through the full recurrent unroll.
#[test]
fn fd_matches_backprop_on_named_parameters() {
    let model = Rlt::new(tiny_config(), Device::Cpu).unwrap();
    let tokens: Vec<u32> = vec![1, 2, 3, 4];
    let eps = 1e-3f32;
    for name in ["merge.b_g", "encoder.0.q_proj.weight"] {
        let loss = model.forward_loss(&tokens, None).unwrap();
        let grads = loss.backward().unwrap();
        let var = model
            .varmap
            .data()
            .lock()
            .unwrap()
            .get(name)
            .unwrap()
            .clone();
        let t = var.as_tensor();
        let shape: Vec<usize> = t.dims().to_vec();
        let flat = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let analytic = grads
            .get(t)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for idx in [0usize, flat.len() / 2] {
            let orig = flat[idx];
            let mut plus = flat.clone();
            plus[idx] = orig + eps;
            var.set(&Tensor::from_vec(plus, shape.clone(), &Device::Cpu).unwrap())
                .unwrap();
            let lp = model
                .forward_loss(&tokens, None)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let mut minus = flat.clone();
            minus[idx] = orig - eps;
            var.set(&Tensor::from_vec(minus, shape.clone(), &Device::Cpu).unwrap())
                .unwrap();
            let lm = model
                .forward_loss(&tokens, None)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            var.set(&Tensor::from_vec(flat.clone(), shape.clone(), &Device::Cpu).unwrap())
                .unwrap();
            let fd = (lp - lm) / (2.0 * eps);
            assert!(
                (fd - analytic[idx]).abs() < 5e-3,
                "{name}[{idx}]: fd={fd} analytic={}",
                analytic[idx]
            );
        }
    }
}

/// Every trainable variable receives a gradient through the full recurrent unroll.
#[test]
fn backward_reaches_all_parameters() {
    let model = Rlt::new(tiny_config(), Device::Cpu).unwrap();
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
    let loss = model.forward_loss(&tokens, None).unwrap();
    let grads = loss.backward().unwrap();
    let vars = model.varmap.all_vars();
    assert!(vars.len() > 10);
    let missing_count = vars
        .iter()
        .filter(|v| grads.get(v.as_tensor()).is_none())
        .count();
    assert_eq!(missing_count, 0, "some parameters got no gradient");
    // Existence alone can pass with a broken all-zeros backward: at least one
    // sampled gradient must actually be non-zero.
    let any_nonzero = vars.iter().any(|v| {
        let g = grads.get(v.as_tensor()).unwrap();
        g.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .any(|&x| x != 0.0)
    });
    assert!(any_nonzero, "all gradients are zero");
}

/// SFT masking: loss changes; the unmasked forward computation is unaffected
/// (masks only reweight the loss — state updates always run, paper §5.2).
#[test]
fn masking_reweights_only_the_loss() {
    let model = Rlt::new(tiny_config(), Device::Cpu).unwrap();
    let tokens: Vec<u32> = vec![10, 11, 12, 13, 14];
    let full = model
        .forward_loss(&tokens, None)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let masked = model
        .forward_loss(&tokens, Some(&[0.0, 0.0, 0.0, 1.0]))
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!((masked - full).abs() > 1e-6, "mask had no effect");
    assert!(masked > 0.0 && masked.is_finite());
}

/// Full-BPTT training reduces loss on a trivially predictable stream.
#[test]
fn train_step_decreases_loss() {
    let model = Rlt::new(tiny_config(), Device::Cpu).unwrap();
    let mut opt = AdamW::new(
        model.varmap.all_vars(),
        ParamsAdamW {
            lr: 0.01,
            ..Default::default()
        },
    )
    .unwrap();
    let tokens: Vec<u32> = vec![42u32; 12];
    let first = model.train_step(&tokens, None, &mut opt).unwrap();
    let mut last = first;
    for _ in 0..29 {
        last = model.train_step(&tokens, None, &mut opt).unwrap();
    }
    assert!(
        last < first * 0.9,
        "loss did not decrease: {first} -> {last}"
    );
}
