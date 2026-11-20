"""Run one experiment config = one results table.

    uv run python -m src.run configs/tcn_ensemble.yaml
    uv run python -m src.run configs/gbm.yaml --subsets FD001 --seeds 0
    uv run python -m src.run configs/tcn_ensemble.yaml --set data.normalization=global --name ablation_global_norm
    uv run python -m src.run configs/tcn_ensemble.yaml --smoke     # quick end-to-end check

Every config is merged on top of configs/base.yaml. Outputs go to results/<name>/:
    runs.csv        one row per (subset, seed, method) with all metrics
    summary.md      mean +- std over seeds for the `report` columns (paste into the paper)
    summary.csv     mean and std of every metric
    predictions/    per-engine test predictions and intervals
    config.yaml     the fully resolved config, meta.json: package versions + git commit
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import uq
from .data import build_dataset
from .decision import decision_metrics
from .evaluate import evaluate_prediction, level_tag

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "base.yaml"


# --- config handling -------------------------------------------------------------------------

def deep_update(base: dict, other: dict | None) -> dict:
    out = copy.deepcopy(base)
    for key, value in (other or {}).items():
        out[key] = deep_update(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def set_value(cfg: dict, assignment: str) -> None:
    """Apply a `--set a.b.c=value` override (value parsed as YAML, e.g. 40, true, [FD001, FD002])."""
    key, sep, raw = assignment.partition("=")
    if not sep:
        raise ValueError(f"--set expects KEY=VALUE, got {assignment!r}")
    *parents, leaf = key.split(".")
    node = cfg
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = yaml.safe_load(raw)


def load_config(path, overrides=(), name=None, subsets=None, seeds=None, smoke=False) -> dict:
    cfg = deep_update(yaml.safe_load(BASE_CONFIG.read_text()), yaml.safe_load(Path(path).read_text()))
    cfg["name"] = name or cfg.get("name") or Path(path).stem
    for assignment in overrides:
        set_value(cfg, assignment)
    if subsets:
        cfg["data"]["subsets"] = list(subsets)
    if seeds:
        cfg["seeds"] = list(seeds)
    if smoke:
        cfg["name"] += "_smoke"
        cfg["seeds"] = cfg["seeds"][:1]
        cfg["data"]["subsets"] = cfg["data"]["subsets"][:1]
        cfg["train"]["epochs"] = 2
        cfg["uq"]["ensemble_size"] = 2
        cfg["uq"]["mc_samples"] = 5
        if cfg["model"]["type"] == "gbm":
            cfg["model"]["n_estimators"] = 30
    return cfg


# --- model runners: each returns ({method: Prediction}, {method: extra metrics}, artifact) ----

def run_neural(data, cfg, seed):
    import torch

    from .models.nn import build_neural  # torch is only imported for deep models

    torch.set_num_threads(cfg["runtime"]["threads"])
    levels = cfg["uq"]["levels"]
    y_scale = float(data.rul_cap or data.train.y.max())
    members = []
    for m in range(cfg["uq"]["ensemble_size"]):
        model = build_neural(cfg["model"], cfg["train"], data.n_features, y_scale, seed=seed * 1000 + m)
        model.fit(data.train.X, data.train.y, data.val.X, data.val.y)
        best = model.history[model.best_epoch - 1] if model.best_epoch else model.history[-1]
        print(f"    member {m}: best epoch {model.best_epoch:3d}, val RMSE {best['rmse']:.2f}, val NLL {best['nll']:.3f}")
        members.append(model)

    single = members[0]
    mu_1, sigma_1 = single.predict_dist(data.test.X)
    cal_mu_1, _ = single.predict_dist(data.cal.X)
    mu_mc, sigma_mc, _ = uq.mc_dropout(single, data.test.X, cfg["uq"]["mc_samples"])
    mu_e, sigma_e, epistemic = uq.ensemble(members, data.test.X)
    cal_mu_e, cal_sigma_e, _ = uq.ensemble(members, data.cal.X)
    q_abs = uq.calibrate(data.cal.y, cal_mu_1, 1.0, levels)
    q_norm = uq.calibrate(data.cal.y, cal_mu_e, cal_sigma_e, levels)

    preds = {
        "gaussian": uq.gaussian_prediction(mu_1, sigma_1, levels),  # one network, aleatoric only
        "mc_dropout": uq.gaussian_prediction(mu_mc, sigma_mc, levels),  # same network, dropout at test time
        "ensemble": uq.gaussian_prediction(mu_e, sigma_e, levels),  # deep ensemble (aleatoric + epistemic)
        "conformal": uq.conformal_prediction(mu_1, 1.0, q_abs),  # split conformal on |residual|
        "ensemble_conformal": uq.conformal_prediction(mu_e, sigma_e, q_norm),  # normalised conformal
    }
    extra = {"ensemble": {"epistemic_share": float(np.mean(epistemic / sigma_e**2))}}
    return preds, extra, (members, q_norm)


def run_gbm(data, cfg, seed):
    from .models.gbm import GBMQuantile

    levels = cfg["uq"]["levels"]
    params = {k: v for k, v in cfg["model"].items() if k != "type"}
    model = GBMQuantile(levels, seed=seed, **params).fit(data.train.X, data.train.y, data.val.X, data.val.y)
    point, bounds = model.predict(data.test.X)
    cal_point, cal_bounds = model.predict(data.cal.X)
    preds = {
        "quantile": uq.quantile_prediction(point, bounds),  # raw quantile regression
        "cqr": uq.cqr(data.cal.y, cal_bounds, point, bounds),  # conformalised quantile regression
        "conformal": uq.conformal_prediction(point, 1.0, uq.calibrate(data.cal.y, cal_point, 1.0, levels)),
    }
    return preds, {}, None


RUNNERS = {"gbm": run_gbm, "tcn": run_neural, "lstm": run_neural}


# --- output ----------------------------------------------------------------------------------

def environment_info() -> dict:
    import lightgbm
    import scipy
    import sklearn

    info = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
            "lightgbm": lightgbm.__version__}
    try:
        import torch
        info["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        info["git_commit"] = git("rev-parse", "HEAD")
        info["git_dirty"] = bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        pass
    return info


def save_predictions(path: Path, windows, pred) -> None:
    df = pd.DataFrame({"unit": windows.unit, "cycle": windows.cycle, "rul_true": windows.rul,
                       "target": windows.y, "point": pred.point})
    for lv, (lo, hi) in sorted(pred.intervals.items()):
        df[f"lo{level_tag(lv)}"], df[f"hi{level_tag(lv)}"] = lo, hi
    if pred.sigma is not None:
        df["sigma"] = pred.sigma
    df.to_csv(path, index=False, float_format="%.4f")


def fmt(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "-"
    digits = 0 if abs(mean) >= 100 else 1 if abs(mean) >= 10 else 3
    return f"{mean:.{digits}f}" if pd.isna(std) else f"{mean:.{digits}f} ± {std:.{digits}f}"


def summarize(runs: pd.DataFrame, report: list[str]):
    metrics = [c for c in runs.columns if c not in ("subset", "seed", "method")]
    grouped = runs.groupby(["subset", "method"], sort=False)[metrics]
    mean, std = grouped.mean(), grouped.std(ddof=1)  # std is NaN with a single seed
    full = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    cols = [c for c in report if c in metrics]
    table = pd.DataFrame({c: [fmt(m, s) for m, s in zip(mean[c], std[c])] for c in cols}, index=mean.index)
    return full, table.reset_index()


def markdown(df: pd.DataFrame) -> str:
    rows = [list(df.columns), ["---"] * len(df.columns), *df.astype(str).values.tolist()]
    return "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)


# --- main ------------------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="experiment YAML, merged on top of configs/base.yaml")
    ap.add_argument("--subsets", nargs="+", help="override data.subsets, e.g. FD001 FD003")
    ap.add_argument("--seeds", nargs="+", type=int, help="override seeds")
    ap.add_argument("--name", help="results folder name (default: config file name)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any value, e.g. data.window=40")
    ap.add_argument("--smoke", action="store_true", help="1 subset, 1 seed, 2 epochs: checks the pipeline end to end")
    ap.add_argument("--threads", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)),
                    help="PyTorch CPU threads (default: min(8, cores/2); more threads are often slower for these small nets)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set, args.name, args.subsets, args.seeds, args.smoke)
    cfg["runtime"] = {"threads": args.threads}
    runner = RUNNERS[cfg["model"]["type"]]
    out = ROOT / "results" / cfg["name"]
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "meta.json").write_text(json.dumps(environment_info(), indent=2))
    data_cfg = {k: v for k, v in cfg["data"].items() if k != "subsets"}

    rows = []
    for subset in cfg["data"]["subsets"]:
        for seed in cfg["seeds"]:
            t0 = time.time()
            print(f"[{cfg['name']}] {subset} seed={seed}")
            data = build_dataset(subset, seed=seed, **data_cfg)
            print(f"    windows: train {len(data.train)}, val {len(data.val)}, cal {len(data.cal)}, "
                  f"test {len(data.test)} engines, {data.n_features} features")
            preds, extra, artifact = runner(data, cfg, seed)
            for method, pred in preds.items():
                rows.append({"subset": subset, "seed": seed, "method": method,
                             **evaluate_prediction(data.test.y, pred, rul_raw=data.test.rul),
                             **decision_metrics(data.test.rul, pred, **cfg["decision"]),
                             **extra.get(method, {})})
                save_predictions(out / "predictions" / f"{subset}_seed{seed}_{method}.csv", data.test, pred)
                r = rows[-1]
                print(f"    {method:<20} RMSE {r['rmse']:6.2f}  NASA {r['nasa']:8.1f}  "
                      f"cov90 {r.get('cov90', float('nan')):.2f}  width90 {r.get('width90', float('nan')):6.1f}")
            if artifact and cfg.get("save_artifact") and seed == cfg["seeds"][0]:
                from .artifact import save_bundle

                members, q_norm = artifact
                save_bundle(ROOT / "artifacts" / cfg["name"] / subset, members, data.preprocessor, cfg["model"],
                            data.window, q_norm, subset)
            pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)  # keep partial results if a later run fails
            print(f"    done in {time.time() - t0:.0f}s")

    full, table = summarize(pd.DataFrame(rows), cfg["report"])
    full.to_csv(out / "summary.csv")
    md = markdown(table)
    (out / "summary.md").write_text(md + "\n", encoding="utf-8")
    print(f"\n{md}\n\nresults written to {out}")


if __name__ == "__main__":
    main()
