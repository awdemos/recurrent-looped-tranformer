//! rlt — CLI for the Recurrent Looped Transformer (rlt-core).

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use candle_nn::Optimizer;
use rlt_core::{load_checkpoint, save_checkpoint, ByteTokenizer, Device, Rlt, RltConfig, Sampler};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Parser)]
#[command(name = "rlt", version, about = "Recurrent Looped Transformer (RLT) CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Write a randomly initialized checkpoint.
    Init {
        /// Output checkpoint path.
        #[arg(long)]
        out: PathBuf,
        /// Model width.
        #[arg(long, default_value_t = 256)]
        d_model: usize,
        /// Encoder = decoder layer count.
        #[arg(long, default_value_t = 4)]
        layers: usize,
        /// Attention heads.
        #[arg(long, default_value_t = 8)]
        heads: usize,
        /// SWA window size.
        #[arg(long, default_value_t = 64)]
        window: usize,
        /// Use untied decoder weights (default: tied, paper §2.6).
        #[arg(long)]
        untied: bool,
    },
    /// Train on a text corpus (byte-level) with full BPTT.
    Train {
        /// Corpus text file.
        #[arg(long)]
        corpus: PathBuf,
        /// Checkpoint to start from (omit to train a fresh default model).
        #[arg(long)]
        init: Option<PathBuf>,
        /// Output checkpoint path.
        #[arg(long)]
        out: PathBuf,
        /// Optimizer steps.
        #[arg(long, default_value_t = 100)]
        steps: usize,
        /// Learning rate.
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        /// Sequence window length (independent segments, paper §5.1).
        #[arg(long, default_value_t = 128)]
        seq_len: usize,
        /// Windows per optimizer step (loss-averaged).
        #[arg(long, default_value_t = 1)]
        batch: usize,
        /// Print loss every N steps.
        #[arg(long, default_value_t = 10)]
        log_every: usize,
    },
    /// Generate text from a prompt.
    Generate {
        /// Checkpoint path.
        #[arg(long)]
        model: PathBuf,
        /// Prompt text.
        #[arg(long)]
        prompt: String,
        /// Maximum tokens to generate.
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        /// Sampling temperature (0 = greedy).
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        /// Optional top-k.
        #[arg(long)]
        top_k: Option<usize>,
        /// RNG seed.
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Init { out, d_model, layers, heads, window, untied } => {
            let cfg = RltConfig {
                d_model,
                n_heads: heads,
                n_encoder_layers: layers,
                n_decoder_layers: layers,
                window,
                tied: !untied,
                ..Default::default()
            };
            cfg.validate().context("invalid config")?;
            let model = Rlt::new(cfg, Device::Cpu)?;
            save_checkpoint(&model, &out)?;
            println!("wrote {}", out.display());
        }
        Commands::Train { corpus, init, out, steps, lr, seq_len, batch, log_every } => {
            let text = std::fs::read_to_string(&corpus)
                .with_context(|| format!("reading {}", corpus.display()))?;
            let tok = ByteTokenizer::new(rlt_core::MIN_VOCAB)?;
            let tokens = tok.encode(&text, true, true);
            let model = match init {
                Some(p) => load_checkpoint(&p, Device::Cpu)?,
                None => Rlt::new(RltConfig::default(), Device::Cpu)?,
            };
            let mut opt = candle_nn::AdamW::new(
                model.varmap.all_vars(),
                candle_nn::ParamsAdamW { lr, ..Default::default() },
            )?;
            let windows: Vec<Vec<u32>> = tokens
                .chunks(seq_len)
                .filter(|c| c.len() >= 2)
                .map(|c| c.to_vec())
                .collect();
            anyhow::ensure!(!windows.is_empty(), "corpus too short for seq-len {seq_len}");
            let mut step = 0;
            while step < steps {
                let group: Vec<&Vec<u32>> =
                    (0..batch).map(|b| &windows[(step + b) % windows.len()]).collect();
                let loss = if group.len() == 1 {
                    model.forward_loss(group[0], None)?
                } else {
                    let mut total = model.forward_loss(group[0], None)?;
                    for w in &group[1..] {
                        total = (total + model.forward_loss(w, None)?)?;
                    }
                    total.affine(1.0 / group.len() as f64, 0.0)?
                };
                let val = loss.to_scalar::<f32>()? as f64;
                opt.backward_step(&loss)?;
                step += 1;
                if step % log_every == 0 || step == 1 || step == steps {
                    println!("step {step}/{steps} loss {val:.4}");
                }
            }
            save_checkpoint(&model, &out)?;
            println!("wrote {}", out.display());
        }
        Commands::Generate { model, prompt, max_tokens, temperature, top_k, seed } => {
            let model = load_checkpoint(&model, Device::Cpu)?;
            let tok = ByteTokenizer::new(rlt_core::MIN_VOCAB)?;
            let prompt_tokens = tok.encode(&prompt, true, false);
            let sampler = if temperature <= 0.0 {
                Sampler::Greedy
            } else {
                Sampler::Sample { temperature: temperature as f32, top_k }
            };
            let mut rng = StdRng::seed_from_u64(seed);
            let (sampled, _) = model.generate(&prompt_tokens, max_tokens, &sampler, &mut rng, true)?;
            let ids: Vec<u32> = sampled.iter().map(|t| t.token).collect();
            let logprob_sum: f64 = sampled.iter().map(|t| t.logprob as f64).sum();
            println!("{}", tok.decode(&ids));
            eprintln!("tokens: {}  logprob sum: {logprob_sum:.3}", ids.len());
        }
    }
    Ok(())
}
