use rlt_core::RltConfig;

#[test]
fn defaults_validate() {
    let cfg = RltConfig::default();
    cfg.validate().unwrap();
    assert_eq!(cfg.head_dim(), cfg.d_model / cfg.n_heads);
}

#[test]
fn rejects_invalid() {
    let cfg = RltConfig {
        n_heads: 3,
        ..Default::default()
    }; // 256 % 3 != 0
    assert!(cfg.validate().is_err());
    let cfg = RltConfig {
        window: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn rejects_degenerate_dimensions() {
    let cfg = RltConfig {
        n_heads: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
    let mut cfg = RltConfig::default();
    cfg.n_decoder_layers = cfg.n_encoder_layers + 1; // tied: L_D > L_E
    assert!(cfg.validate().is_err());
    let cfg = RltConfig {
        d_model: 18,
        n_heads: 2,
        ..Default::default()
    };
    assert!(cfg.validate().is_err()); // head_dim 9 is odd
    let cfg = RltConfig {
        feedback_alpha: f64::NAN,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn memory_group_mapping() {
    let cfg = RltConfig {
        memory_groups: 3,
        n_decoder_layers: 7,
        ..Default::default()
    };
    assert_eq!(cfg.memory_group_of(0), 0);
    assert_eq!(cfg.memory_group_of(5), 2);
    assert_eq!(cfg.memory_group_of(6), 0);
}
