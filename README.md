# Alg-ProNC

Code for a study of Neural-Collapse-style geometry in analytic (closed-form)
class-incremental learning: a frozen pretrained backbone feeds a ridge-regression
head that is updated recursively via Sherman-Morrison-Woodbury, with Simplex-ETF
regression targets in place of one-hot targets.

The paper's two headline results:

1. **An impossibility result.** For a ridge head on frozen features, replacing
   one-hot targets with a Simplex ETF changes classifier scores only by a fixed
   positive rescale plus a per-class constant, so **argmax is unchanged** —
   Simplex-ETF targets are provably a no-op for standard analytic
   class-incremental learning. This is verified numerically to ~1e-14 on real
   features across ridge strengths and backbones (`exp0_equivalence`).
2. **An equinorm remedy for imbalanced streams.** The invariance above only
   covers *linear* re-targeting. Applying a *non-linear* post-process to the
   fitted classifier weights (equalizing column norms, or projecting onto the
   nearest Simplex ETF) breaks the invariance and materially recovers
   tail-class accuracy under long-tailed class-incremental streams, at a small
   cost on balanced streams (`exp3_imbalance`).

All experiments after feature extraction are CPU linear algebra and run in
seconds; only the one-time feature-extraction step benefits from a GPU.

## Setup

```bash
pip install -e ".[dev]"
# or, with uv:
uv sync
```

Requires Python >= 3.10. CIFAR-100 and TinyImageNet are downloaded
automatically on first use (into `./data` by default, or `$ALGPRONC_DATA_DIR`
if set). Extracted backbone features are cached to
`$ALGPRONC_DATA_DIR/features/<hash>.pt` so each (dataset, backbone) pair is
only run through the backbone once.

Run the test suite with:

```bash
pytest -q
```

## Reproducing the paper's results

Every command below writes its output under `./runs/<run_id>/` (`summary.json`
/ `tasks.jsonl`), overridable with `--set out_dir=...`. `--set a.b=c` overrides
any config field from the command line without editing the YAML.

### Result 1 — target-geometry invariance and the staleness artifact (`exp0_equivalence`)

Exact argmax-agreement between one-hot and Simplex-ETF targets, and the
accuracy collapse of the literal (non-retro-corrected) ETF-expansion
algorithm:

```bash
python -m algpronc.experiments.exp0_equivalence --config configs/cifar100_t10.yaml
```

Sweep the ridge strength to confirm the invariance holds at every `lambda`:

```bash
python -m algpronc.experiments.exp0_equivalence --config configs/cifar100_t10.yaml --set cl.ridge_lambda=<lambda>
# lambda in {1e-4, 1e-3, 1e-2, 1e-1, 1.0}
```

Repeat on a second backbone to de-risk backbone-specificity:

```bash
python -m algpronc.experiments.exp0_equivalence --config configs/cifar100_t10.yaml \
  --set model.backbone=resnet50 --set model.feature_dim=2048
```

### Result 2 — main class-incremental benchmark table (`exp1_main`)

```bash
python -m algpronc.experiments.exp1_main --config configs/cifar100_t10.yaml
python -m algpronc.experiments.exp1_main --config configs/cifar100_t20.yaml
python -m algpronc.experiments.exp1_main --config configs/tinyimagenet_t10.yaml
```

Repeat with `--set model.backbone=resnet50 --set model.feature_dim=2048` for
the ResNet-50 robustness rows.

### Result 3 — compressed-frame sweep, `m < K-1` (`exp2_compressed`)

```bash
python -m algpronc.experiments.exp2_compressed --config configs/cifar100_t10.yaml \
  --set 'exp2.frames=[welch, random, etf]'
```

For the 5-seed table, repeat with `--set seed=<s>` for `s in 0..4`.

### Result 4 — equinorm remedy under class imbalance (`exp3_imbalance`)

Single ratio, clean `onehot` vs `equinorm:onehot` ablation:

```bash
python -m algpronc.experiments.exp3_imbalance --config configs/cifar100_longtail.yaml
```

Full imbalance-ratio sweep with error bars (8 ratios x 5 seeds):

```bash
python -m algpronc.experiments.exp3_imbalance --config configs/cifar100_longtail.yaml \
  --set cl.imbalance.imbalance_ratio=<r> --set seed=<s> --set cl.imbalance.seed=<s>
# r in {1, 2, 5, 10, 25, 50, 100, 200}, s in 0..4
```

Four-head ladder (`onehot`, `equinorm:onehot`, `equiangular:onehot`,
`etf_project:onehot`) used for the equinorm-vs-equiangular ablation:

```bash
python -m algpronc.experiments.exp3_imbalance --config configs/cifar100_longtail.yaml \
  --set 'cl.heads=[onehot, equinorm:onehot, equiangular:onehot, etf_project:onehot]' \
  --set cl.imbalance.imbalance_ratio=<r> --set seed=<s> --set cl.imbalance.seed=<s>
```

Repeat any of the above with `--set model.backbone=resnet50 --set model.feature_dim=2048`
and/or `--set data.dataset=tinyimagenet` for the full backbone x dataset
transfer matrix.

### Random-projection dimension sweep (`exp4_rp`)

```bash
python -m algpronc.experiments.exp4_rp --config configs/cifar100_t10.yaml \
  --set 'exp4.proj_dims=[0, 1000, 5000, 10000]'
```

Note: `proj_dim=10000` is expensive (the float64 SMW recursion over a
10000x10000 covariance takes tens of minutes on CPU); the smaller dims are
fast.

## Layout

```
configs/*.yaml     one config per experiment; override any field with --set a.b=c
src/algpronc/
  geometry/frames.py   Simplex ETF, the compared frame constructions, diagnostics
  heads/               the analytic head and its onehot / etf / compressed / equinorm variants
  data/                class-incremental splits, long-tail sampling, datasets, feature caching
  models/              frozen backbones, random-projection expansion
  engine.py            the task-stream loop; fits every head on identical features
  experiments/exp*.py  the runnable entrypoints above, one per experiment
tests/               unit tests, including the numerical checks behind the paper's claims
```
