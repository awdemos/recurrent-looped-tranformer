# rlt-cli

CLI for the [Recurrent Looped Transformer](https://github.com/yifanzhang-pro/recurrent-looped-tranformer)
(`rlt-core`).

## Install

```bash
cargo install rlt-cli
```

## Usage

```bash
# random checkpoint (defaults: --d-model 256 --layers 4 --heads 8 --window 64)
rlt init --out model.st

# train on a corpus (byte-level, full BPTT, independent seq-len segments)
rlt train --corpus corpus.txt --init model.st --out model.st --steps 1000 --lr 1e-3 --seq-len 128

# generate (temperature 0 = greedy)
rlt generate --model model.st --prompt "To be" --max-tokens 64 --temperature 0.8 --seed 0
```

## License

MIT OR Apache-2.0.
