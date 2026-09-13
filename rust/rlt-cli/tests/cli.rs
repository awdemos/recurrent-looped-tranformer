use std::path::PathBuf;
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_rlt")
}

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rlt_cli_test_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn init_train_generate_roundtrip() {
    let dir = tmpdir("rtg");
    let model = dir.join("model.st");
    let corpus = dir.join("corpus.txt");
    std::fs::write(
        &corpus,
        "the quick brown fox jumps over the lazy dog. ".repeat(20),
    )
    .unwrap();

    let init = Command::new(bin())
        .args(["init", "--out"])
        .arg(&model)
        .args(["--d-model", "64", "--layers", "2", "--heads", "4"])
        .output()
        .unwrap();
    assert!(
        init.status.success(),
        "init failed: {}",
        String::from_utf8_lossy(&init.stderr)
    );
    assert!(
        model.is_file(),
        "init did not write a checkpoint to {}",
        model.display()
    );

    let train = Command::new(bin())
        .args(["train", "--corpus"])
        .arg(&corpus)
        .args(["--init"])
        .arg(&model)
        .args(["--out"])
        .arg(&model)
        .args(["--steps", "3", "--seq-len", "48"])
        .output()
        .unwrap();
    assert!(
        train.status.success(),
        "train failed: {}",
        String::from_utf8_lossy(&train.stderr)
    );

    let gen = Command::new(bin())
        .args(["generate", "--model"])
        .arg(&model)
        .args(["--prompt", "the", "--max-tokens", "8", "--temperature", "0"])
        .output()
        .unwrap();
    assert!(
        gen.status.success(),
        "generate failed: {}",
        String::from_utf8_lossy(&gen.stderr)
    );
    let stdout = String::from_utf8_lossy(&gen.stdout);
    assert!(!stdout.trim().is_empty(), "generate produced no output");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn train_with_batch_two_succeeds() {
    let dir = tmpdir("batch");
    let corpus = dir.join("corpus.txt");
    std::fs::write(
        &corpus,
        "the quick brown fox jumps over the lazy dog. ".repeat(20),
    )
    .unwrap();
    let out = dir.join("model.st");

    let train = Command::new(bin())
        .args(["train", "--corpus"])
        .arg(&corpus)
        .args(["--out"])
        .arg(&out)
        .args(["--steps", "2", "--batch", "2", "--seq-len", "48"])
        .output()
        .unwrap();
    assert!(
        train.status.success(),
        "train --batch 2 failed: {}",
        String::from_utf8_lossy(&train.stderr)
    );
    assert!(
        out.is_file(),
        "train did not write a checkpoint to {}",
        out.display()
    );

    std::fs::remove_dir_all(&dir).ok();
}
