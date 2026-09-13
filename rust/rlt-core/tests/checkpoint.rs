use candle_core::Device;
use rlt_core::{load_checkpoint, save_checkpoint, Rlt, RltConfig};

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

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("rlt_test_{}_{}.st", name, std::process::id()))
}

#[test]
fn checkpoint_roundtrip_preserves_logits() {
    let path = tmp("roundtrip");
    let cfg = tiny_config();
    let model = Rlt::new(cfg.clone(), Device::Cpu).unwrap();
    let tokens = vec![3u32, 1, 4, 1, 5, 9];
    let (before, _) = model.prefill(&tokens).unwrap();
    save_checkpoint(&model, &path).unwrap();
    let loaded = load_checkpoint(&path, Device::Cpu).unwrap();
    assert_eq!(loaded.config, cfg);
    let (after, _) = loaded.prefill(&tokens).unwrap();
    for (a, b) in before.iter().zip(after.iter()) {
        let d = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(d, 0.0, "logits changed across checkpoint");
    }
    std::fs::remove_file(&path).ok();
    std::fs::remove_file(path.with_extension("st.json")).ok();
}

#[test]
fn load_missing_config_errors() {
    let path = tmp("missing");
    assert!(load_checkpoint(&path, Device::Cpu).is_err());
}
