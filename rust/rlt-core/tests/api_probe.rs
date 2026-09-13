//! Compile+run probe for every candle API this crate depends on.
//!
//! Adapted for candle 0.11 (deviations from the plan's original probe):
//! - `sigmoid`/`softmax`/`log_softmax` are free fns in `candle_nn::ops`, not `Tensor` methods.
//! - `Init::Uniform` field is `up` (renamed from `hi`).
//! - `VarBuilder::{get,get_with_hints}` return `Tensor` (not `Var`), so no `.as_tensor()`.
//!   Fetch the underlying `Var` from `VarMap::data()` to call `Var::set`.
//! - `Optimizer::new(vars, config)`: `AdamW::new(vm.all_vars(), ParamsAdamW { lr, ..Default::default() })`.
//! - `GradStore::get` takes `&Tensor` directly.
//! - `to_scalar` requires rank 0 (the plan called it on a (2,2) tensor); probed with a scalar tensor.
//!
//! 4-D extraction: reshape to 3-D + `to_vec3` (`to_vec4` does not exist in candle).
//!
//! Coverage beyond the plan's probe:
//! - Value assertions: softmax rows sum to ~1, gather fetches the expected elements,
//!   AdamW decreases the recomputed loss.
//! - `VarMap::save`/`VarMap::load` safetensors roundtrip via a temp file.
//! - One-liners: `Var::from_tensor`, `broadcast_mul`, `gelu`, `squeeze`, `contiguous`,
//!   `abs`, `max_all`, `sum(dim)`, `to_vec1`, `dims3`, `dims`, `flatten_all`, `&x + &x`, `&x * 2.0`.
use candle_core::{D, Device, DType, Tensor, Var};
use candle_nn::ops;
use candle_nn::{AdamW, Embedding, Init, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

fn dev() -> Device { Device::Cpu }

#[test]
fn tensor_ops_probe() -> candle_core::Result<()> {
    let d = dev();
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &d)?;
    let _ = x.powf(2.)?.mean(D::Minus1)?.sqrt()?;
    let _ = x.broadcast_add(&Tensor::new(1e-5f32, &d)?)?;
    let _ = ops::sigmoid(&x.affine(2.0, 1.0)?)?;
    let _ = x.matmul(&x)?;
    let _ = ops::softmax(&x, D::Minus1)?;
    let _ = ops::log_softmax(&x, D::Minus1)?;
    let idx = Tensor::from_vec(vec![1u32, 0], (2, 1), &d)?;
    let _ = x.gather(&idx, 1)?;
    let _ = x.narrow(0, 0, 1)?;
    let _ = Tensor::cat(&[&x, &x], 0)?;
    let _ = Tensor::stack(&[&x, &x], 0)?;
    let _ = Tensor::ones((2, 2), DType::F32, &d)?;
    let _ = Tensor::arange(0u32, 4, &d)?.to_dtype(DType::F32)?.cos()?.sin()?;
    let _ = x.to_vec2::<f32>()?;
    let _ = x.unsqueeze(0)?;
    let _ = x.transpose(0, 1)?;
    let _ = x.reshape((4,))?;
    let _ = x.sum_all()?;
    let _ = x.broadcast_div(&Tensor::new(2.0f32, &d)?)?;
    let _ = x.dim(D::Minus1)?;
    let _ = x.dims2()?;
    assert_eq!(Tensor::new(1.0f32, &d)?.to_scalar::<f32>()?, 1.0);
    let _ = Tensor::new(vec![vec![0f32, f32::NEG_INFINITY], vec![0.0, 0.0]], &d)?;
    let _ = Tensor::rand(-1f32, 1f32, (2, 3), &d)?;

    // --- extra op coverage (one-liners) ---
    let _ = x.broadcast_mul(&x)?;
    let _ = x.gelu()?;
    let _ = x.contiguous()?;
    let _ = x.abs()?;
    let _ = x.max_all()?;
    let _ = x.sum(D::Minus1)?; // dim-sum, distinct from sum_all
    let _ = x.flatten_all()?;
    let _ = x.dims();
    let _ = x.reshape((4,))?.to_vec1::<f32>()?;
    let _ = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &d)?.squeeze(0)?.to_vec1::<f32>()?;
    // tensor arithmetic via std::ops impls
    let _ = (&x + &x)?;
    let _ = (&x * 2.0)?;

    // --- 4-D extraction (Tasks 6/9) ---
    // 4-D extraction: reshape to 3-D + `to_vec3` (`to_vec4` does not exist in candle).
    let x4 = Tensor::from_vec((0..24).map(|i| i as f32).collect::<Vec<_>>(), (1, 2, 3, 4), &d)?;
    let (a, b, c, d2) = x4.dims4()?;
    let x3 = x4.reshape((a * b, c, d2))?;
    let _ = x3.dims3()?;
    let v3 = x3.to_vec3::<f32>()?;
    assert_eq!(v3.len(), 2);
    assert_eq!(v3[0][0], [0.0, 1.0, 2.0, 3.0]);

    // --- value assertions: catch semantic breakage, not just missing APIs ---
    // softmax rows sum to ~1
    let s = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &d)?;
    let sm = ops::softmax(&s, D::Minus1)?;
    for rs in sm.sum(D::Minus1)?.to_vec1::<f32>()? {
        assert!((rs - 1.0).abs() < 1e-6, "softmax row sum = {rs}");
    }
    // gather returns the source elements at the gathered indices:
    // x = [[1,2],[3,4]], idx = [[1],[0]] -> [[2],[3]]
    assert_eq!(x.gather(&idx, 1)?.to_vec2::<f32>()?, [[2.0], [3.0]]);
    Ok(())
}

#[test]
fn var_and_optimizer_probe() -> candle_core::Result<()> {
    let d = dev();
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &d);
    let w = vb.get_with_hints((2, 2), "w", Init::Uniform { lo: -0.1, up: 0.1 })?;
    let lin = Linear::new(w.clone(), None);
    let emb = Embedding::new(vb.get((4, 2), "emb")?, 2);
    let b = vb.pp("merge").get((2,), "b")?;
    assert_eq!(b.dims1()?, 2);
    // VarMap::data() must be public for gradient-check tests
    let _names: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
    // Var::set must exist (finite-difference tests mutate parameters);
    // VarBuilder::get returns Tensor in 0.11, so pull the Var out of the map.
    let vars = vm.data().lock().unwrap();
    let b_var: &Var = vars.get("merge.b").unwrap();
    b_var.set(&Tensor::new(&[0.5f32, 0.25], &d)?)?;
    drop(vars);

    // VarMap::save/load roundtrip via a temp safetensors file.
    let path = std::env::temp_dir().join(format!("rlt_api_probe_{}.safetensors", std::process::id()));
    vm.save(&path)?;
    let mut vm2 = VarMap::new();
    let vb2 = VarBuilder::from_varmap(&vm2, DType::F32, &d);
    // load writes into existing vars, so pre-create the same names in the fresh map
    let b2 = vb2.pp("merge").get((2,), "b")?;
    vm2.load(&path)?;
    assert_eq!(b2.to_vec1::<f32>()?, [0.5f32, 0.25]);
    std::fs::remove_file(&path)?;

    let ids = Tensor::from_vec(vec![0u32, 1, 2], (3,), &d)?;
    let _ = emb.forward(&ids)?;
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &d)?;
    let _v = Var::from_tensor(&x)?;
    let y = lin.forward(&x)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 0.01, ..Default::default() })?;
    let loss = y.sqr()?.sum_all()?;
    let loss_before = loss.to_scalar::<f32>()?;
    opt.backward_step(&loss)?;
    // AdamW must decrease the loss when recomputed with the updated weights
    let loss_after = lin.forward(&x)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
    assert!(loss_after < loss_before, "AdamW did not decrease loss: {loss_before} -> {loss_after}");
    let grads = loss.backward()?;
    let _g = grads.get(&w);
    Ok(())
}
