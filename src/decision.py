"""Cost-based maintenance decision on top of RUL predictions (snapshot policy).

At an inspection we either maintain now or wait until the next opportunity in `horizon` cycles:

    maintain now  if  statistic <= horizon

`statistic` is the point prediction (risk-neutral policy) or the lower bound of a prediction
interval (risk-averse policy: maintain as soon as failure within the horizon is plausible).

Cost per engine:
    maintain now                    c_pm + c_waste * true_rul   (planned stop + unused life thrown away)
    wait, engine fails in horizon   c_fail                      (true_rul < horizon)
    wait, engine survives           0

Evaluated once per test engine at its last observed cycle, i.e. the same protocol as the RMSE,
using the true (uncapped) RUL. `cost_oracle` (decide with the true RUL) is the best achievable cost.
"""
from __future__ import annotations

import numpy as np

from .evaluate import level_tag


def maintenance_cost(rul_true, statistic, horizon: float = 20, c_pm: float = 1.0, c_fail: float = 10.0,
                     c_waste: float = 0.01) -> dict:
    rul_true = np.asarray(rul_true, dtype=float)
    maintain = np.asarray(statistic) <= horizon
    failure = ~maintain & (rul_true < horizon)
    cost = np.where(maintain, c_pm + c_waste * rul_true, np.where(failure, c_fail, 0.0))
    return {"cost": float(cost.mean()), "failures": int(failure.sum()), "maintained": int(maintain.sum()),
            "wasted_cycles": float((rul_true * maintain).sum())}


def decision_metrics(rul_true, pred, **costs) -> dict:
    """Mean cost, failures and maintenance actions for the point policy and each lower-bound policy."""
    policies = {"point": pred.point, **{f"lo{level_tag(lv)}": lo for lv, (lo, _) in sorted(pred.intervals.items())}}
    out = {}
    for name, statistic in policies.items():
        res = maintenance_cost(rul_true, statistic, **costs)
        out[f"cost_{name}"] = res["cost"]
        out[f"fail_{name}"] = res["failures"]
        out[f"maint_{name}"] = res["maintained"]
    out["cost_oracle"] = maintenance_cost(rul_true, rul_true, **costs)["cost"]
    return out
