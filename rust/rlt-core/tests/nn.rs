use candle_core::{Device, Tensor, D};
use rlt_core::nn;
use rlt_core::nn::RotaryEmbedding;

fn dev() -> Device {
    Device::Cpu
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

#[test]
fn rms_norm_matches_manual() {
    let x = Tensor::new(&[[3.0f32, 4.0], [1.0, 2.0]], &dev()).unwrap();
    let y = nn::rms_norm(&x, 1e-5).unwrap().to_vec2::<f32>().unwrap();
    let ms0: f32 = (9.0 + 16.0) / 2.0;
    for (got, want) in y[0].iter().zip([3.0 / ms0.sqrt(), 4.0 / ms0.sqrt()]) {
        assert!((got - want).abs() < 1e-5);
    }
}

#[test]
fn rope_known_angles() {
    // dim=4, base=10000: pair angles at pos 1 are [1.0, 0.01]
    let rotary = RotaryEmbedding::new(4, 8, &dev()).unwrap();
    let x = Tensor::new(&[[[[1.0f32, 0.0, 1.0, 0.0]]]], &dev()).unwrap();
    let (q, _) = rotary.apply_qk(&x, &x, 1).unwrap();
    let v = t4(&q);
    let (c, s) = (1.0f32.cos(), 1.0f32.sin());
    assert!(
        (v[0][0][0][0] - (c - s)).abs() < 1e-5,
        "got {}",
        v[0][0][0][0]
    );
    assert!(v[0][0][0][1].abs() < 1e-6);
    assert!((v[0][0][0][2] - (c + s)).abs() < 1e-5);
    assert!(v[0][0][0][3].abs() < 1e-6);
}

#[test]
fn rope_preserves_norm() -> candle_core::Result<()> {
    let rotary = RotaryEmbedding::new(8, 16, &dev())?;
    let x = Tensor::rand(-1.0f32, 1.0, (1, 2, 5, 8), &dev())?;
    let (q, _) = rotary.apply_qk(&x, &x, 3)?;
    let a = x.powf(2.)?.sum(D::Minus1)?.sqrt()?;
    let b = q.powf(2.)?.sum(D::Minus1)?.sqrt()?;
    let diff = (a - b)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-4);
    Ok(())
}

#[test]
fn sdpa_matches_naive() {
    let d = dev();
    let q = Tensor::rand(-1.0f32, 1.0, (1, 2, 3, 4), &d).unwrap();
    let k = Tensor::rand(-1.0f32, 1.0, (1, 2, 5, 4), &d).unwrap();
    let v = Tensor::rand(-1.0f32, 1.0, (1, 2, 5, 4), &d).unwrap();
    let got = t4(&nn::sdpa(&q, &k, &v, None).unwrap());
    let qv = t4(&q);
    let kv = t4(&k);
    let vv = t4(&v);
    let scale = 1.0 / 4.0f32.sqrt();
    for h in 0..2 {
        for i in 0..3 {
            let mut scores = [0f32; 5];
            for j in 0..5 {
                let mut dot = 0f32;
                for x in 0..4 {
                    dot += qv[0][h][i][x] * kv[0][h][j][x];
                }
                scores[j] = dot * scale;
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            for x in 0..4 {
                let mut want = 0f32;
                for j in 0..5 {
                    want += (exps[j] / z) * vv[0][h][j][x];
                }
                assert!((got[0][h][i][x] - want).abs() < 1e-4);
            }
        }
    }
}

#[test]
fn causal_mask_blocks_future() {
    let m = nn::causal_mask(3, &dev()).unwrap();
    let v = t4(&m);
    assert_eq!(v[0][0][0][1], f32::NEG_INFINITY);
    assert_eq!(v[0][0][0][2], f32::NEG_INFINITY);
    assert_eq!(v[0][0][1][2], f32::NEG_INFINITY);
    assert_eq!(v[0][0][2][0], 0.0);
    assert_eq!(v[0][0][0][0], 0.0);
}

#[test]
fn sdpa_with_causal_mask() {
    let d = dev();
    let q = Tensor::rand(-1.0f32, 1.0, (1, 1, 3, 4), &d).unwrap();
    let k = Tensor::rand(-1.0f32, 1.0, (1, 1, 3, 4), &d).unwrap();
    let v = Tensor::rand(-1.0f32, 1.0, (1, 1, 3, 4), &d).unwrap();
    let mask = nn::causal_mask(3, &d).unwrap();
    let got = t4(&nn::sdpa(&q, &k, &v, Some(&mask)).unwrap());
    let qv = t4(&q);
    let kv = t4(&k);
    let vv = t4(&v);
    let scale = 1.0 / 4.0f32.sqrt();
    for i in 0..3 {
        // Naive masked reference: scores above the diagonal are -inf, so row i
        // attends only to keys 0..=i.
        let mut scores = [0f32; 3];
        for j in 0..3 {
            let mut dot = 0f32;
            for x in 0..4 {
                dot += qv[0][0][i][x] * kv[0][0][j][x];
            }
            scores[j] = if j > i {
                f32::NEG_INFINITY
            } else {
                dot * scale
            };
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let z: f32 = exps.iter().sum();
        for x in 0..4 {
            let mut want = 0f32;
            for j in 0..=i {
                want += (exps[j] / z) * vv[0][0][j][x];
            }
            assert!((got[0][0][i][x] - want).abs() < 1e-4);
        }
    }
    // Explicit causality: row 0 has a single valid key, so its output is v[0].
    for x in 0..4 {
        assert!((got[0][0][0][x] - vv[0][0][0][x]).abs() < 1e-5);
    }
}
