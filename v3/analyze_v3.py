"""Frozen V3 observed-energy analysis. Read prediction products; never fit models.

All QLIKE values use physical bp^2, and all confidence intervals are exploratory
conditional on saved forecasts. The six registered pairs share annual draws.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = ("Mean", "SMA20", "EWMA", "OwnHAR", "MultiHAR", "Fixed", "Uniform", "Content")
NEURAL = ("Fixed", "Uniform", "Content")
PAIRS = (("Content", "Uniform"), ("Content", "Fixed"), ("MultiHAR", "OwnHAR"),
         ("OwnHAR", "EWMA"), ("MultiHAR", "EWMA"), ("Fixed", "MultiHAR"))
PRIMARY = PAIRS[:2]
YEARS = (2022, 2023, 2024)
COUNTS = {2022: 238, 2023: 238, 2024: 113}
SEEDS = (202611051, 202611052, 202611053)
BOOTSTRAP_SEED = 20261105
BLOCKS = (5, 10, 20)
REPLICATIONS = 2000
KEYS = ["year", "index", "date"]
META = ["information_end", "label_end", "q5"]

def qlike(q, f):
    q = np.asarray(q, dtype=float)
    f = np.asarray(f, dtype=float)
    if q.shape != f.shape or not np.isfinite(q).all() or not np.isfinite(f).all():
        raise ValueError("Aligned finite targets and forecasts required")
    if (q < 0).any() or (f <= 0).any():
        raise ValueError("Keep Q=0; require Q>=0 and forecast>0")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        loss = np.log(f) + q / f
    if not np.isfinite(loss).all():
        raise ValueError("Nonfinite loss: do not remove or floor observations")
    return loss

def paired_difference(q, model_f, baseline_f):
    q = np.asarray(q, dtype=float)
    mf = np.asarray(model_f, dtype=float)
    bf = np.asarray(baseline_f, dtype=float)
    # Separate logarithms avoid overflow from an intermediate mf / bf.
    lm = qlike(q, mf)
    lb = qlike(q, bf)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        delta = np.log(mf) - np.log(bf) + q / mf - q / bf
    np.testing.assert_allclose(delta, lm - lb, rtol=1e-10, atol=1e-10)
    return delta

def metrics(q, f):
    q = np.asarray(q, float); f = np.asarray(f, float)
    loss = qlike(q, f)
    return dict(n=len(q), qlike=float(loss.mean()), mae=float(np.abs(f-q).mean()),
                mse=float(np.square(f-q).mean()),
                log1p_mse=float(np.square(np.log1p(f)-np.log1p(q)).mean()),
                mean_actual=float(q.mean()), mean_forecast=float(f.mean()),
                zero_targets=int((q == 0).sum()))

def matrix_statistics(values, years):
    """Annual equal-weight macro and request-weight pooled, never seed-weight."""
    v = np.asarray(values, float); y = np.asarray(years)
    if v.ndim == 1: v = v[:, None]
    if len(v) != len(y) or not np.isfinite(v).all():
        raise ValueError("Finite aligned matrix required")
    labels = sorted(set(y))
    annual = {int(year): v[y == year].mean(axis=0) for year in labels}
    result = {"macro": np.stack(list(annual.values())).mean(axis=0),
              "pooled": v.mean(axis=0)}
    result.update({str(year): value for year, value in annual.items()})
    return result

def moving_block_indices(n, block, rng):
    """Noncircular overlapping blocks; uniform starts; concatenate then trim."""
    if block < 1 or n < block:
        raise ValueError("Need n>=block>=1")
    starts = rng.integers(0, n-block+1, size=(n+block-1)//block)
    return (starts[:, None] + np.arange(block)).reshape(-1)[:n]

def joint_bootstrap(values, years, block, replications=REPLICATIONS,
                    seed=BOOTSTRAP_SEED):
    """Every matrix column (model/pair) uses the same annual sampled indices.

    Each block length resets the same seed. Within each replicate years are
    drawn in ascending order. No blocks cross years or wrap at year ends.
    """
    v = np.asarray(values, float); y = np.asarray(years)
    if v.ndim == 1: v = v[:, None]
    point = matrix_statistics(v, y)
    labels = sorted(set(y)); groups = [np.flatnonzero(y == year) for year in labels]
    rng = np.random.default_rng(seed)
    result = {key: np.empty((replications, v.shape[1])) for key in point}
    for iteration in range(replications):
        annual = []
        for year, ix in zip(labels, groups):
            selected = ix[moving_block_indices(len(ix), block, rng)]
            mean = v[selected].mean(axis=0)
            annual.append(mean); result[str(year)][iteration] = mean
        result["macro"][iteration] = np.stack(annual).mean(axis=0)
        result["pooled"][iteration] = np.average(np.stack(annual), axis=0,
                                               weights=[len(ix) for ix in groups])
    return result

def interval_table(values, years, pairs=PAIRS, blocks=BLOCKS,
                   replications=REPLICATIONS):
    point = matrix_statistics(values, years); rows = []
    for block in blocks:
        draws = joint_bootstrap(values, years, block, replications)
        for j, (model, baseline) in enumerate(pairs):
            for statistic in point:
                for level in ([.95, .975] if (model, baseline) in PRIMARY and statistic == "macro" else [.95]):
                    alpha = (1-level)/2
                    lower, upper = np.quantile(draws[statistic][:, j], [alpha, 1-alpha])
                    rows.append(dict(model=model, baseline=baseline,
                        statistic=statistic, block=block, confidence=level,
                        estimate=float(point[statistic][j]), lower=float(lower),
                        upper=float(upper), replications=replications,
                        valid_replications=int(np.isfinite(draws[statistic][:, j]).sum()),
                        bootstrap_seed=BOOTSTRAP_SEED))
    return pd.DataFrame(rows)

def evaluation_rows(frame):
    frame = frame.copy()
    if "role" in frame:
        # Reject mixed roles instead of silently mixing or discarding products.
        if set(frame.role) != {"evaluation"}:
            raise ValueError("Expected evaluation-only input; TRAIN/VALIDATION stay separate")
    required = set(KEYS + META + ["model", "forecast"])
    if required - set(frame):
        raise ValueError("Missing columns: " + str(sorted(required-set(frame))))
    for col in ["index", "year", "q5", "forecast"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    for col in ["date", "information_end", "label_end"]:
        parsed = pd.to_datetime(frame[col], errors="raise")
        if parsed.isna().any(): raise ValueError("Missing date")
        frame[col] = parsed.dt.strftime("%Y-%m-%d")
    if (frame.information_end >= frame.date).any() or (frame.label_end < frame.date).any():
        raise ValueError("Invalid information/maturity ordering")
    if (frame.label_end > "2024-06-30").any():
        raise ValueError("Sealed period violation")
    if (pd.to_datetime(frame.label_end).dt.year != frame.year).any():
        raise ValueError("Evaluation labels must mature within their registered year")
    if (pd.to_datetime(frame.date).dt.year != frame.year).any():
        raise ValueError("Year disagrees with request date")
    if not np.isfinite(frame["index"]).all() or not np.equal(frame["index"], np.floor(frame["index"])).all():
        raise ValueError("Noninteger request index")
    qlike(frame.q5.to_numpy(), frame.forecast.to_numpy())
    return frame

def align_ensemble(frame, enforce_counts=True):
    f = evaluation_rows(frame)
    if set(f.model) != set(MODELS):
        raise ValueError("Exactly eight registered models required")
    if f.duplicated(KEYS+["model"]).any():
        raise ValueError("Duplicate model/request key")
    ref = f[f.model == "Mean"].sort_values(KEYS).reset_index(drop=True)
    if not ref["index"].is_monotonic_increasing or not ref.date.is_monotonic_increasing:
        raise ValueError("Original request index/date order not monotone")
    if ref.duplicated(["index"]).any() or ref.duplicated(["date"]).any():
        raise ValueError("Request index/date not one-to-one")
    if enforce_counts:
        if ref.groupby("year").size().to_dict() != COUNTS or len(ref) != 589:
            raise ValueError("Expected fixed 238/238/113=589 request ledger")
    forecasts = []
    for name in MODELS:
        item = f[f.model == name].sort_values(KEYS).reset_index(drop=True)
        if len(item) != len(ref) or not item[KEYS+META].equals(ref[KEYS+META]):
            raise ValueError("Different request/target ledger for " + name)
        forecasts.append(item.forecast.to_numpy())
    return ref, np.stack(forecasts, axis=1)

def validate_seeds(frame, reference, forecasts):
    f = evaluation_rows(frame)
    if "seed" not in f: raise ValueError("Seed column required")
    if set(f.model) != set(MODELS): raise ValueError("Seed models differ")
    if f.duplicated(KEYS+["model", "seed"]).any(): raise ValueError("Duplicate seed key")
    rows = []
    for j, name in enumerate(MODELS):
        group = f[f.model == name]
        expected = set(SEEDS) if name in NEURAL else {0}
        if set(group.seed) != expected: raise ValueError("Unexpected seeds for "+name)
        predicted = []
        for seed in sorted(expected):
            item = group[group.seed == seed].sort_values(KEYS).reset_index(drop=True)
            if len(item) != len(reference) or not item[KEYS+META].equals(reference[KEYS+META]):
                raise ValueError("Seed ledger mismatch")
            predicted.append(item.forecast.to_numpy())
            annual = []
            for year in YEARS:
                selected = item.year == year
                m = metrics(item.loc[selected, "q5"], item.loc[selected, "forecast"])
                annual.append(m)
                rows.append(dict(model=name, seed=seed, aggregation="annual", year=year, **m))
            rows.append(dict(model=name, seed=seed, aggregation="macro", year="all",
                             n=len(item), **{k: float(np.mean([m[k] for m in annual]))
                             for k in annual[0] if k not in ("n", "zero_targets")},
                             zero_targets=int((item.q5 == 0).sum())))
        # Arithmetic average of physical forecasts, NOT a geometric/log mean.
        np.testing.assert_allclose(np.stack(predicted).mean(0), forecasts[:, j],
                                   rtol=1e-12, atol=1e-10)
    return pd.DataFrame(rows)

def describe_models(reference, forecasts):
    q = reference.q5.to_numpy(); y = reference.year.to_numpy()
    annual = []; model_rows = []
    for j, name in enumerate(MODELS):
        per_year = []
        for year in YEARS:
            ix = y == year; row = metrics(q[ix], forecasts[ix, j])
            per_year.append(row); annual.append(dict(model=name, year=year, **row))
        pooled = metrics(q, forecasts[:, j])
        macro = {k: float(np.mean([r[k] for r in per_year]))
                 for k in pooled if k not in ("n", "zero_targets")}
        model_rows.append(dict(model=name, aggregation="macro", n=len(q),
                               zero_targets=int((q == 0).sum()), **macro))
        model_rows.append(dict(model=name, aggregation="pooled", **pooled))
    return pd.DataFrame(model_rows), pd.DataFrame(annual)

def tail_sensitivity(reference, forecasts):
    q = reference.q5.to_numpy(); y = reference.year.to_numpy(); rows = []
    for year in YEARS:
        ix = np.flatnonzero(y == year)
        # Stable order breaks equal Q by earlier date (reference sorted by date).
        ordered = ix[np.argsort(-q[ix], kind="stable")]
        k = int(np.ceil(.01*len(ix))); tail = ordered[:k]; largest = ordered[:1]
        keep = np.setdiff1d(ix, tail); keep_one = np.setdiff1d(ix, largest)
        for model, baseline in PAIRS:
            mf = forecasts[:, MODELS.index(model)]; bf = forecasts[:, MODELS.index(baseline)]
            delta = paired_difference(q, mf, bf)
            rows.append(dict(year=year, model=model, baseline=baseline, n=len(ix),
                n_tail=k, tail_dates=";".join(reference.iloc[tail].date),
                largest_q_date=reference.iloc[largest[0]].date,
                q_share=float(q[tail].sum()/q[ix].sum()) if q[ix].sum()>0 else None,
                full_delta=float(delta[ix].mean()),
                tail_mean_delta=float(delta[tail].mean()),
                tail_contribution_to_full_mean=float(delta[tail].sum()/len(ix)),
                without_top1pct_delta=float(delta[keep].mean()),
                without_largest_q_delta=float(delta[keep_one].mean()),
                tail_model_qlike=float(qlike(q[tail], mf[tail]).mean()),
                tail_baseline_qlike=float(qlike(q[tail], bf[tail]).mean()),
                interpretation="ex_post_description_only_no_selection"))
    return pd.DataFrame(rows)

def continuation_decision(intervals, year_deltas, model_metrics):
    tests = []
    for model, baseline in PRIMARY:
        row = intervals[(intervals.model==model)&(intervals.baseline==baseline)&
                        (intervals.statistic=="macro")&(intervals.block==10)&
                        np.isclose(intervals.confidence, .975)]
        if len(row)!=1: raise ValueError("Missing unique primary 97.5 macro interval")
        r = row.iloc[0]
        ys = year_deltas[(year_deltas.model==model)&(year_deltas.baseline==baseline)]
        negatives = int((ys.delta<0).sum())
        passed = bool(r.estimate<0 and r.upper<0 and negatives>=2 and len(ys)==3)
        tests.append(dict(model=model, baseline=baseline, macro_delta=float(r.estimate),
                          upper_97_5=float(r.upper), negative_years=negatives, passed=passed))
    lookup = model_metrics[model_metrics.aggregation=="macro"].set_index("model").qlike
    financial = {name: float(lookup["Content"]-lookup[name])
                 for name in ("Mean", "SMA20", "EWMA", "OwnHAR", "MultiHAR")}
    return dict(mechanism_signal=all(t["passed"] for t in tests), primary_checks=tests,
                content_minus_financial_macro_point_differences=financial,
                financial_significance_tested=False,
                interpretation="NN mechanism signal is not proof of financial advantage; "
                "retrospective exposed data, fixed finite protocol, no automatic new fits")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(run_dir, output):
    run_dir = Path(run_dir); output = Path(output)
    if output.exists(): raise FileExistsError("Analysis output must be a new directory")
    status_path = run_dir/"status.json"
    if not status_path.exists() or json.loads(status_path.read_text(encoding="utf-8"))["status"] != "COMPLETE":
        raise ValueError("Training run must declare status COMPLETE before analysis")
    if (run_dir/"FAILED.json").exists():
        raise ValueError("Run has a failure record; do not analyze as COMPLETE")
    ensemble_path=run_dir/"predictions_ensemble.csv"
    seed_path=run_dir/"predictions_seeds.csv"
    reference, forecasts = align_ensemble(pd.read_csv(ensemble_path))
    seed_metrics = validate_seeds(pd.read_csv(seed_path), reference, forecasts)
    model_metrics, year_metrics = describe_models(reference, forecasts)
    q=reference.q5.to_numpy(); y=reference.year.to_numpy()
    deltas=np.column_stack([paired_difference(q, forecasts[:, MODELS.index(m)],
                                             forecasts[:, MODELS.index(b)]) for m,b in PAIRS])
    intervals=interval_table(deltas,y)
    year_rows=[]
    for j,(model,baseline) in enumerate(PAIRS):
        for year in YEARS:
            ix=y==year
            year_rows.append(dict(model=model,baseline=baseline,year=year,
                                  n=int(ix.sum()),delta=float(deltas[ix,j].mean())))
    annual_deltas=pd.DataFrame(year_rows)
    tail=tail_sensitivity(reference,forecasts)
    decision=continuation_decision(intervals,annual_deltas,model_metrics)
    # Full point differences versus financial baselines are descriptive, not
    # additional paired-bootstrap significance searches.
    points=[]
    for name in MODELS:
        for baseline in ("Mean","SMA20","EWMA","OwnHAR","MultiHAR"):
            if name==baseline:continue
            d=paired_difference(q,forecasts[:,MODELS.index(name)],forecasts[:,MODELS.index(baseline)])
            for stat,value in matrix_statistics(d,y).items():
                points.append(dict(model=name,baseline=baseline,statistic=stat,delta=float(value[0])))
    manifest=dict(analysis_source_sha256=sha(__file__),
        ensemble_sha256=sha(ensemble_path),seeds_sha256=sha(seed_path),
        real_fits=0,eval_requests=589,eval_counts=COUNTS,models=MODELS,pairs=PAIRS,
        seed_ensemble="arithmetic mean of physical forecasts",
        loss="physical log(f)+Q/f; no posthoc floors or target exclusions",
        bootstrap=dict(seed=BOOTSTRAP_SEED,blocks=BLOCKS,replications=REPLICATIONS,
            circular=False,start_range="inclusive 0..n-B",concatenate="ceil(n/B) blocks then trim n",
            draw_order="each block resets seed; replicate outer loop, years ascending",
            joint_models_and_pairs=True),
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        evidence="retrospective_exploratory_previously_exposed_history")
    output.mkdir(parents=True,exist_ok=False)
    for name,frame in dict(model_metrics=model_metrics,year_metrics=year_metrics,
        seed_metrics=seed_metrics,paired_intervals=intervals,year_pair_deltas=annual_deltas,
        tail_sensitivity=tail,financial_point_differences=pd.DataFrame(points)).items():
        frame.to_csv(output/(name+".csv"),index=False)
    for name,value in [("continuation_decision",decision),("analysis_manifest",manifest)]:
        (output/(name+".json")).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return decision

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    if not args.output.resolve().is_relative_to(root/"outputs"):
        raise ValueError("Output must remain within V3 outputs")
    print(json.dumps(run(args.run,args.output),indent=2,ensure_ascii=False))
