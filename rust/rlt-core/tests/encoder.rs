use candle_core::Device;
use rlt_core::{Rlt, RltConfig};

fn tiny_config() -> RltConfig {
    RltConfig {
        d_model: 32,
        n_heads: 2,
        d_ff: 64,
        n_encoder_layers: 2,
        n_decoder_layers: 2,
        window: 4,
        memory_groups: 1,
        vocab_size: 258,
        tied: false,
        feedback_alpha: 0.1,
        max_seq_len: 64,
    }
}

#[test]
fn encode_parallel_matches_incremental() {
    let cfg = tiny_config();
    let model = Rlt::new(cfg.clone(), Device::Cpu).unwrap();
    let tokens: Vec<u32> = vec![5, 9, 1, 2, 3, 8, 7, 6, 4, 0];
    let (e, enc_cache, _mem) = model.encode(&tokens).unwrap();
    assert_eq!(e.dims2().unwrap(), (10, 32));
    let mut state_enc: Vec<Option<rlt_core::LayerKv>> = vec![None; cfg.n_encoder_layers];
    let mut e_steps = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        let e_t = model.encode_step(t, pos, &mut state_enc).unwrap();
        e_steps.push(e_t.squeeze(0).unwrap());
    }
    let e_inc = candle_core::Tensor::stack(&e_steps, 0).unwrap();
    let diff = (e - e_inc)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        diff < 1e-4,
        "parallel vs incremental encoder mismatch: {diff}"
    );
    for c in enc_cache.iter().flatten() {
        assert_eq!(c.len(), tokens.len());
    }
}

#[test]
fn memory_key_value_projections_are_distinct() {
    let model = Rlt::new(tiny_config(), Device::Cpu).unwrap();
    let data = model.varmap.data().lock().unwrap();
    let k = data.get("mem.0.k.weight").expect("mem.0.k.weight missing");
    let v = data.get("mem.0.v.weight").expect("mem.0.v.weight missing");
    assert_ne!(
        k.as_tensor().to_vec2::<f32>().unwrap(),
        v.as_tensor().to_vec2::<f32>().unwrap(),
        "K and V memory projections must be distinct parameters"
    );
}
