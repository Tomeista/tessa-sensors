# Uncertainty-aware RUL prediction on NASA C-MAPSS

Intelligence Engineering project. We predict the remaining useful life (RUL) of turbofan engines with
**prediction intervals** instead of point estimates, compare uncertainty methods side by side, and test whether
calibrated intervals lead to cheaper maintenance decisions than point predictions.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/getting-started/installation/) (Windows: `winget install astral-sh.uv`,
macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`). uv installs Python 3.12 and the exact package
versions pinned in `uv.lock`, so everyone runs identical code.

```bash
uv sync                                                     # .venv with all pinned dependencies
uv run python data/download.py                              # C-MAPSS -> data/raw/ (SHA-256 checked, not in git)
uv run pytest                                               # unit tests, a few seconds
uv run python -m src.run configs/tcn_ensemble.yaml --smoke  # end-to-end check, < 1 min
```

Add packages with `uv add <package>` (never `pip install`), so `pyproject.toml` and `uv.lock` stay in sync.

## Reproducing the results

One command per results table. Each run covers FD001–FD004 × 3 seeds:

| Table | Command | Runtime (laptop CPU) |
|---|---|---|
| GBM baseline | `uv run python -m src.run configs/gbm.yaml` | ~4 min |
| TCN deep ensemble (main) | `uv run python -m src.run configs/tcn_ensemble.yaml` | ~6–7 h (FD001: ~17 min per seed; FD002/FD004 are ~3× larger) |
| LSTM challenger | `uv run python -m src.run configs/lstm_ensemble.yaml` | similar to TCN |

Outputs in `results/<name>/`: `summary.md` (mean ± std over seeds, ready for the paper), `runs.csv` (every
metric per subset/seed/method), `summary.csv`, per-engine predictions, the fully resolved `config.yaml` and
`meta.json` (package versions + git commit). Useful flags: `--subsets FD001`, `--seeds 0`, `--name`,
`--threads N`, `--set key=value`. Long runs can be split across machines or terminals by subset.

**Ablations** = same config, one value changed:

```bash
uv run python -m src.run configs/tcn_ensemble.yaml --set data.normalization=global --name abl_global_norm
uv run python -m src.run configs/tcn_ensemble.yaml --set data.window=50 --name abl_window50
uv run python -m src.run configs/tcn_ensemble.yaml --set uq.ensemble_size=10 --name abl_ensemble10
```

## Pipeline

```
data/download.py -> src/data.py -> src/models/ -> src/uq.py -> src/evaluate.py, src/decision.py -> results/
                    shared         gbm|tcn|lstm   intervals     metrics, maintenance cost
                                                                src/artifact.py -> serve/ (FastAPI + Docker)
```

### Data protocol: identical for every model (`src/data.py`, `configs/base.yaml`)

- **Target:** RUL capped at 125 cycles (piecewise-linear target). The test target is capped the same way
  (`cap_test`); the RMSE against the uncapped true RUL is reported as `rmse_uncapped`.
- **Inputs:** the 14 informative sensors (s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21).
- **Normalisation:** z-score per operating condition. FD002/FD004 have 6 conditions, found with k-means on the
  3 operating settings; those subsets also get the condition one-hot as input. Fitted on training engines only.
- **Windows:** 30 cycles, stride 1, label = RUL at the window's last cycle. Test engines shorter than 30 cycles
  are front-padded with their first row.
- **Splits by engine, never by window:** 70 % train / 15 % validation (early stopping) / 15 % calibration
  (conformal). The split changes with the seed, so mean ± std covers split variance too.
- **Test protocol:** one window per test engine, ending at its last observed cycle (official `RUL_FD00x.txt`).

### Models (`src/models/`)

| Role | Model | Question it answers |
|---|---|---|
| Baseline | LightGBM on window statistics (last, mean, std, min, max, slope) + quantile models | Is deep learning worth it at all? |
| Main | TCN (4 residual blocks, dilations 1-2-4-8) with Gaussian head (μ, σ), β-NLL, deep ensemble of 5 | Accuracy and uncertainty in one model |
| Challenger | LSTM (2 × 64) with the same head, training and UQ | Established reference |

A further challenger (e.g. a frozen time-series foundation model as encoder) only needs a backbone module,
see `src/models/__init__.py`. Ensembles, MC dropout and conformal calibration then work unchanged.

### Uncertainty methods (`src/uq.py`)

All variants come from the same trained networks, so differences are due to the UQ method only:

| Method | What it is |
|---|---|
| `gaussian` | one network's predicted σ (aleatoric uncertainty only) |
| `mc_dropout` | the same network, 50 stochastic forward passes with dropout on |
| `ensemble` | 5 networks (different seeds), mixture moment-matched: aleatoric + epistemic |
| `conformal` | split conformal on the absolute residuals of one network |
| `ensemble_conformal` | normalised split conformal: scores \|y − μ\| / σ of the ensemble |
| `quantile`, `cqr` (GBM) | raw quantile regression, and conformalised quantile regression |

### Metrics (`src/evaluate.py`, `src/decision.py`)

- **Point:** RMSE, MAE, NASA score (sum over engines; late predictions are penalised harder than early ones).
- **Intervals** at 50/80/90/95 %: coverage (`cov90` should be ≈ 0.90), mean width, Winkler interval score,
  calibration error (mean |coverage − level|), NLL and CRPS for the Gaussian methods.
- **Decision:** snapshot maintenance policy, "maintain now if the statistic ≤ 20 cycles", with the point
  prediction (`cost_point`) or a lower interval bound (`cost_lo90`, ...) as statistic. Costs: planned
  maintenance 1, failure 10, wasted life 0.01 per cycle; `cost_oracle` uses the true RUL. All in `configs/base.yaml`.

### Known caveats (for the uncertainty and evaluation workstreams)

- Conformal calibration uses all windows of the calibration engines. Windows of one engine are correlated and
  are not distributed exactly like the test engines' last windows, so the coverage guarantee is approximate.
- Deep models train on 70 % of the training engines (the rest is validation/calibration), so RMSEs are slightly
  worse than in papers that train on all engines. The comparison between our models stays fair.
- Capped target vs. CQR: many calibration windows have true RUL = 125 = the upper quantile prediction, so their
  CQR score is exactly 0. If more than 10 % of the scores are 0, the 90 % correction is 0 and CQR returns the raw
  quantile intervals (seen on FD003 in the first GBM run). An asymmetric CQR (separate lower/upper correction)
  would avoid this.

## Deployment (`serve/`)

```bash
uv run python -m src.run configs/tcn_ensemble.yaml --subsets FD001 --seeds 0   # writes artifacts/tcn_ensemble/FD001/
uv run uvicorn serve.app:app --port 8000                                        # docs at http://localhost:8000/docs
docker build -f serve/Dockerfile -t ie-rul . && docker run -p 8000:8000 ie-rul
```

`POST /predict` with `{"rows": [[op1, op2, op3, s1, ..., s21], ...], "level": 0.9}` (one engine's history,
oldest cycle first) returns the RUL, a calibrated interval (deep ensemble + normalised conformal) and σ.

## Repository layout and workstreams

| Path | Content | Workstream |
|---|---|---|
| `data/download.py` | download + hash check | Data & preprocessing |
| `src/data.py`, `configs/base.yaml` (`data`) | loading, labels, normalisation, windows, splits | Data & preprocessing |
| `src/models/` | GBM, TCN, LSTM, training loop | Modelling |
| `src/uq.py` | ensembles, MC dropout, conformal, CQR | Uncertainty |
| `src/evaluate.py`, `src/decision.py` | metrics, maintenance cost | Evaluation & decision |
| `src/run.py`, `src/artifact.py`, `serve/`, `tests/`, README | experiment runner, deployment, reproducibility | Reproducibility & paper |
| `notebooks/` | exploration only, no result may depend on it | everyone |

## Use of AI tools (log for the paper's AI usage statement)

| Date | Tool | Used for | Review status |
|---|---|---|---|
| 2026-11-15 | Claude | brainstorming the project concept | draft for group discussion |
| 2026-11-18 | Claude Code | initial pipeline code (data, models, UQ, evaluation, serving, tests) | to be reviewed |
