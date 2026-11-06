"""Frozen V5 decision experiment. No network and no access to original market files."""
from __future__ import annotations

import os
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import subprocess

import numpy as np
import pandas as pd
import torch
from torch import nn
from data_v5 import load_market, matrix

ROOT = Path(__file__).resolve().parent
A = np.array([1., 0., -1.])
B = np.array([0., 1., -1.])
TRAD = ("DCC", "Prior", "Uniform", "EWMA", "Kernel", "HAR_CCC", "GARCH_CCC")
PAIRS = TRAD + ("TrainingFixed", "Ridge", "MemoryKernel", "MemoryRecency", "MemoryUniform", "MLP", "ContentNoQuery")
ENTRIES = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump(path, data):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)


def components(h):
    return A @ h @ A, A @ h @ B, B @ h @ B


def action(xx, xy):
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.divide(xy, xx, out=np.full_like(xx, .60), where=xx > 0)
    return np.clip(value, 0., 1.)


def local(u0, change, cap):
    return np.clip(u0 + np.clip(change, -cap, cap), 0., 1.)


def risk(c, u):
    xx, xy, yy = c
    loss = yy - 2 * u * xy + u*u*xx
    if np.any(loss < -1e-9) or not np.isfinite(loss).all():
        raise ValueError("Invalid decision risk")
    return np.maximum(loss, 0.)


def feature(h, slope):
    d = np.diagonal(h, axis1=1, axis2=2)
    return np.stack([h[:, i, j] / np.sqrt(d[:, i]*d[:, j])
                     for i, j in ((0, 1), (0, 2), (1, 2))] +
                    [np.log(d[:, 0] / d[:, 1]), np.log(d[:, 2] / d[:, 1]),
                     np.log(slope / d.sum(axis=1))], axis=1)


def masks(df):
    dates = df.date
    v_start = dates[dates.dt.year == 2019].min()
    tr = (dates.dt.year == 2018) & (df.label_end < v_start) & (df.role == "validation")
    va = (dates.dt.year == 2019) & (df.label_end < pd.Timestamp("2020-01-01")) & (df.role == "validation")
    if not tr.any() or not va.any() or df.loc[tr, "label_end"].max() >= df.loc[va, "date"].min():
        raise ValueError("Invalid V5 train and validation maturity")
    return np.flatnonzero(tr), np.flatnonzero(va)


def history_positions(df, memory):
    # Input is one market, reset integer index and contiguous date-ordered folds.
    result = []
    for _, group in df.groupby("fold", sort=False):
        rows = group.index.to_numpy()
        assert np.array_equal(group.date.to_numpy(), np.sort(group.date.to_numpy()))
        date = group.date.to_numpy()
        end = group.label_end.to_numpy()
        assert np.array_equal(end, np.sort(end))
        for t in date:
            stop = np.searchsorted(end, t, side="left")
            result.append(rows[max(0, stop-memory):stop][::-1])
    return result


def prepare(df, mean, sd, cfg):
    h = matrix(df, "DCC")
    hx, hy, _ = components(h)
    if np.any(hx <= 0):
        raise ValueError("DCC slope energy must be positive")
    u0 = action(hx, hy)
    comp = components(matrix(df))
    q = np.clip((feature(h, hx)-mean)/sd, -cfg["normalization_clip"], cfg["normalization_clip"])
    rows = history_positions(df, cfg["memory"])
    n, width = len(df), cfg["memory"]
    keys = np.zeros((n, width, 6))
    vals = np.zeros((n, width, 2))
    mask = np.zeros((n, width), dtype=bool)
    for i, idx in enumerate(rows):
        if len(idx) < cfg["min_memory"]:
            mask[i, 0] = True  # Null slot, with zero values and zero correction.
            continue
        keys[i, :len(idx)] = q[idx]
        vals[i, :len(idx)] = np.column_stack([comp[1][idx]/hx[idx], comp[0][idx]/hx[idx]])
        mask[i, :len(idx)] = True
    count = np.array(list(map(len, rows)))
    if not np.isfinite(vals).all() or np.any(vals[:, :, 1] < -1e-9):
        raise ValueError("Invalid mature case values")
    return dict(q=q, k=keys, vals=vals, mask=mask, u0=u0, comp=comp, count=count)


def tensors(pack, idx):
    return [torch.from_numpy(pack[k][idx]) for k in ("q", "k", "vals", "mask", "u0")]


def scores_action(logit, v, mask, u0, cfg):
    weights = torch.softmax(logit.masked_fill(~mask, -torch.inf), dim=1)
    sums = (weights.unsqueeze(2)*v).sum(dim=1)
    change = (sums[:, 0] - u0*sums[:, 1]) / (sums[:, 1] + cfg["local_penalty"])
    return torch.clamp(u0 + torch.clamp(change, -cfg["local_cap"], cfg["local_cap"]), 0., 1.)


def recency(k, cfg):
    return -cfg["age_decay"]*torch.arange(k.shape[1], dtype=k.dtype)


class Scorer(nn.Module):
    def __init__(self, kind, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.query = nn.Parameter(torch.randn(6, 4, generator=gen, dtype=torch.float64)*.2)
        self.key = nn.Parameter(torch.zeros(6, 4, dtype=torch.float64))
        self.age = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.kind = kind

    def forward(self, q, k, v, mask, u0, cfg, no_query=False):
        if no_query:
            q = torch.zeros_like(q)
        q_proj, k_proj = (q@self.query).unsqueeze(1), k@self.key
        score = (q_proj*k_proj).sum(dim=2)/2 if self.kind == "Content" else torch.tanh(q_proj+k_proj).sum(dim=2)/2
        age = recency(k, cfg)
        return scores_action(score + age*(1 + self.age), v, mask, u0, cfg)


def rule(name, q, k, v, mask, u0, cfg):
    score = recency(k, cfg).expand(q.shape[0], -1)
    if name == "MemoryUniform":
        score = torch.zeros_like(score)
    if name == "MemoryKernel":
        score = score - (q.unsqueeze(1)-k).square().sum(dim=2)/12
    return scores_action(score, v, mask, u0, cfg)


class Budget:
    def __init__(self, out, limit):
        self.path = out/"fit_budget.jsonl"
        self.limit = limit
        self.claimed = set()

    def event(self, **row):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False)+"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def fit(self, job, operation):
        if job in self.claimed or len(self.claimed) >= self.limit:
            raise RuntimeError("Spent fit budget")
        self.event(event="claim", job=job)
        self.claimed.add(job)
        start = time.perf_counter()
        try:
            obj = operation()
        except BaseException as error:
            self.event(event="finish", job=job, status="failed", seconds=time.perf_counter()-start,
                       error=f"{type(error).__name__}: {error}")
            raise
        self.event(event="finish", job=job, status="complete", seconds=time.perf_counter()-start)
        return obj


def fit_model(kind, seed, pack, tr, va, scale, cfg, out):
    model = Scorer(kind, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    fit_x, val_x = tensors(pack, tr), tensors(pack, va)
    fit_y, val_y = ([torch.from_numpy(x[idx]) for x in pack["comp"]] for idx in (tr, va))
    def loss(x, y):
        u = model(*x, cfg)
        return ((y[2] - 2*u*y[1] + u*u*y[0])/scale).mean()

    best = (math.inf, 0, {})
    path = out/f"trace_{kind}_{seed}.csv"
    with path.open("x", encoding="utf-8") as handle:
        handle.write("epoch,training_normalized_risk,validation_normalized_risk,gradient_norm\n")
        for epoch in range(1, cfg["epochs"]+1):
            opt.zero_grad()
            loss(fit_x, fit_y).backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"]))
            opt.step()
            with torch.no_grad():
                lt, lv = float(loss(fit_x, fit_y)), float(loss(val_x, val_y))
            if not all(map(math.isfinite, (norm, lt, lv))):
                raise ValueError("Nonfinite model trajectory")
            handle.write(f"{epoch},{lt:.17g},{lv:.17g},{norm:.17g}\n")
            handle.flush()
            if lv < best[0]:
                best = (lv, epoch, {name: w.detach().clone() for name, w in model.state_dict().items()})
    model.load_state_dict(best[2])
    torch.save(best[2], out/f"weights_{kind}_{seed}.pt")
    return model, dict(model=kind, seed=seed, epoch=best[1], normalized_validation=best[0],
                       parameters=sum(p.numel() for p in model.parameters()))


def outcomes(df, pack, models, fixed, ridge, cfg):
    idx = np.flatnonzero(df.role == "evaluation")
    e = df.iloc[idx].copy()
    comp = tuple(v[idx] for v in pack["comp"])
    u0 = pack["u0"][idx]
    cases = tensors(pack, idx)
    acts = {(name, 0): action(*components(matrix(e, name))[:2]) for name in TRAD}
    acts["Geometry", 0] = np.full(len(e), .60)
    acts["TrainingFixed", 0] = np.full(len(e), fixed)
    acts["Ridge", 0] = local(u0, np.column_stack([np.ones(len(e)), pack["q"][idx]])@ridge, cfg["local_cap"])
    with torch.no_grad():
        for name in ("MemoryRecency", "MemoryUniform", "MemoryKernel"):
            acts[name, 0] = rule(name, *cases, cfg).numpy()
        for name in ("MLP", "Content", "ContentNoQuery"):
            values = []
            base = "Content" if name == "ContentNoQuery" else name
            for seed in cfg["seeds"]:
                values.append(models[base, seed](*cases, cfg, no_query=name=="ContentNoQuery").numpy())
                acts[name, seed] = values[-1]
            acts[name, 0] = np.mean(values, axis=0)

    ids = ("market", "fold", "period", "role", "index", "date", "information_end", "label_end",
           "y00", "y01", "y02", "y11", "y12", "y22")
    return pd.concat([e[list(ids)].assign(model=name, seed=seed, u0=u0, u=u,
                       risk=risk(comp, u), mature_count=pack["count"][idx])
                      for (name, seed), u in acts.items()], ignore_index=True)


def summarize(results, cfg):
    base = results[results.seed == 0]
    metrics = [{"market": a, "period": str(b), "model": c, "n": len(g), "risk": float(g.risk.mean())}
               for (a, b, c), g in results.groupby(["market", "period", "model"], sort=False)
               if g.seed.eq(0).all()]
    for (a, b, c), g in base.groupby(["market", "period", "model"], sort=False):
        if c in ("Content", "MLP", "ContentNoQuery"):
            metrics.append(dict(market=a, period=str(b), model=c, n=len(g), risk=float(g.risk.mean())))
    seed_metrics = [{"market": a, "period": str(b), "model": c, "seed": int(d), "n": len(g), "risk": float(g.risk.mean())}
                    for (a,b,c,d), g in results[results.seed != 0].groupby(["market","period","model","seed"], sort=False)]
    pairs = []
    period_pairs = []
    sensitivity = []
    for block in cfg["blocks"]:
        rng = np.random.default_rng(cfg["bootstrap_seed"]+block)
        for market in ("us", "china"):
            point, draws = [], []
            periods = base[base.market == market].groupby("period", sort=False)
            for period, group in periods:
                w = group.pivot(index="date", columns="model", values="risk")
                arr = w[["Content", *PAIRS]].to_numpy()
                if len(arr) <= block or not np.isfinite(arr).all():
                    raise ValueError("Incomplete paired periods")
                means = arr.mean(axis=0)
                point.append(np.divide(means[0], means[1:], where=means[1:] > 0,
                                       out=np.full_like(means[1:], np.nan)) - 1)
                if block == cfg["blocks"][0]:
                    period_pairs.extend(dict(market=market, period=str(period), baseline=m, point=float(x) if np.isfinite(x) else None)
                                        for m,x in zip(PAIRS, point[-1]))
                    for offset in range(5):
                        means5 = arr[offset::5].mean(axis=0)
                        sensitivity.extend(dict(market=market, period=str(period), offset=offset, baseline=m,
                                                point=float(means5[0]/d - 1) if d>0 else None)
                                           for m,d in zip(PAIRS, means5[1:]))
                count = math.ceil(len(arr)/block)
                starts = rng.integers(0, len(arr)-block+1, size=(cfg["replicates"], count))
                sample = (starts[:, :, None] + np.arange(block)).reshape(cfg["replicates"], -1)[:, :len(arr)]
                avg = arr[sample].mean(axis=1)
                ratio = np.divide(avg[:, :1], avg[:, 1:], where=avg[:, 1:] > 0,
                                  out=np.full_like(avg[:, 1:], np.nan))
                draws.append(ratio-1)
            pct = np.mean(draws, axis=0)
            for j, name in enumerate(PAIRS):
                x = pct[:, j]
                q = np.quantile(x, [.025, .975, 1-.05/len(PAIRS)]) if np.isfinite(x).all() else [None]*3
                effect = float(np.mean(point, axis=0)[j])
                pairs.append(dict(market=market, baseline=name, block=block,
                                  point=effect if math.isfinite(effect) else None,
                                  lower=q[0], upper=q[1], family_upper=q[2],
                                  negative_periods=int((np.array(point)[:,j]<0).sum())))
    return metrics, seed_metrics, pairs, period_pairs, sensitivity


def check_freeze():
    proof = load(ROOT/"FREEZE_MANIFEST.json")
    for name, sha in proof["sha256"].items():
        if digest(ROOT/name) != sha:
            raise RuntimeError(f"Frozen item changed: {name}")
    return proof


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    freeze = check_freeze()
    cfg = load(ROOT/"frozen_config.json")
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    out = ROOT/"outputs/v5_run"
    out.mkdir(exist_ok=False)   # No restart or silent budget reset.
    start = time.perf_counter()
    b = Budget(out, cfg["fit_limit"])
    try:
        dev = load_market(ROOT, "us", roles=("validation",)).reset_index(drop=True)
        tr, va = masks(dev)
        hd = matrix(dev, "DCC")
        f = feature(hd, components(hd)[0])
        mu, sd = f[tr].mean(axis=0), f[tr].std(axis=0)
        if np.any(sd <= 0):
            raise ValueError("Constant development feature")
        pack = prepare(dev, mu, sd, cfg)
        oracle = local(pack["u0"], action(*pack["comp"][:2])-pack["u0"], cfg["local_cap"])
        dcc_risk = risk(pack["comp"], pack["u0"])
        scale = float(dcc_risk[tr].mean())
        if not math.isfinite(scale) or scale <= 0 or float(dcc_risk[va].mean()) <= 0:
            raise ValueError("Invalid development DCC risk denominator")
        oracle_gain = 1 - float(risk(pack["comp"], oracle)[va].mean()/dcc_risk[va].mean())
        valid = int((pack["count"][va]>=cfg["min_memory"]).sum())
        accepted = oracle_gain >= cfg["screen_oracle_improvement"] and valid >= cfg["screen_min_requests"]
        screening = dict(train=len(tr), validation=len(va), memory_ready=valid, val_oracle_improvement=oracle_gain,
                         accepted=bool(accepted), stage="development only", real_fits=0)
        dump(out/"development_screen.json", screening)
        if not accepted:
            dump(out/"summary.json", dict(status="SCREEN_STOP", screen=screening, real_fits=0))
            return
        xx, xy, _ = [z[tr] for z in pack["comp"]]
        fixed = float(action(np.array([xx.sum()]), np.array([xy.sum()]))[0])
        design = np.column_stack([np.ones(len(tr)), pack["q"][tr]])
        ridge = b.fit("Ridge", lambda: np.linalg.solve(
            (design.T*xx)@design/(scale*len(tr)) + cfg["ridge"]*np.diag([0,1,1,1,1,1,1]),
            design.T@(xy-xx*pack["u0"][tr])/(scale*len(tr))))
        if not np.isfinite(ridge).all():
            raise ValueError("Invalid Ridge estimate")
        models, checkpoints = {}, []
        for kind in ("Content", "MLP"):
            for seed in cfg["seeds"]:
                model, info = b.fit(f"{kind}_{seed}", lambda: fit_model(kind, seed, pack, tr, va, scale, cfg, out))
                models[kind, seed] = model
                checkpoints.append(info)
        decisions, identities = [], {}
        for market in ("us", "china"):
            df = load_market(ROOT, market).reset_index(drop=True)
            p = prepare(df, mu, sd, cfg)
            if market == "us":
                ii, jj = masks(df)
                assert np.array_equal(p["q"][ii], pack["q"][tr])
                assert np.array_equal(p["vals"][jj], pack["vals"][va])
            decisions.append(outcomes(df, p, models, fixed, ridge, cfg))
            identities[market] = df.attrs["provenance"]
        decisions = pd.concat(decisions, ignore_index=True)
        decisions.to_csv(out/"decisions.csv", index=False, float_format="%.17g")
        metrics, seeds, pairs, periods, sensitivity = summarize(decisions, cfg)
        main = [p for p in pairs if p["market"]=="us" and p["block"]==cfg["blocks"][0]]
        advance = all(p["point"] is not None and p["point"] <= -cfg["advance_improvement"] and p["family_upper"] is not None
                      and p["family_upper"] < 0 and p["negative_periods"] >= 2 for p in main)
        report = dict(status="COMPLETE", evidence="Exploration and historical transfer", real_fits=len(b.claimed),
                      no_independent_data=True, screen=screening, checkpoints=checkpoints, advance=advance,
                      paired_family=list(PAIRS), metrics=metrics, seed_metrics=seeds, pairs=pairs,
                      period_pairs=periods, five_offset_sensitivity=sensitivity, inputs=identities,
                      shared_training_transforms=2, versions={"numpy":np.__version__,"pandas":pd.__version__,
                          "torch":torch.__version__}, seconds=time.perf_counter()-start,
                      freeze_sha256=digest(ROOT/"FREEZE_MANIFEST.json"), training_commit=commit)
        dump(out/"summary.json", report)
        print(json.dumps({"status":"COMPLETE", "real_fits":len(b.claimed), "advance":advance,
                          "screen":screening}, ensure_ascii=False))
    except BaseException as exc:
        dump(out/"FAILED.json", dict(real_fits=len(b.claimed), error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
