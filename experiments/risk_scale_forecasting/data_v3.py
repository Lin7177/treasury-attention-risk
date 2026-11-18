"""V3 point-in-time Q5 loader; owned implementation, never imports older runners."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ("yield_1y", "yield_3y", "yield_5y", "yield_7y", "yield_10y")
EXPECTED_SOURCE_SHA256 = None
CUTOFF = pd.Timestamp("2024-06-30")
YEARS = (2022, 2023, 2024)
EXPECTED_COUNTS = {2022: (1120, 126, 238), 2023: (1362, 126, 238), 2024: (1604, 126, 113)}

def build_dataset(dates, levels):
    """Build on the unchanged original axis; only authorized caller-supplied levels."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    y = np.asarray(levels, dtype=np.float64)
    if y.shape != (len(dates), 5):
        raise ValueError("Expected original-axis levels [T,5].")
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be unique, ordered and finite.")
    if len(dates) and dates[-1] > CUTOFF:
        raise ValueError("Unapproved dates must be truncated before numeric conversion.")
    with np.errstate(invalid="ignore", over="ignore"):
        delta = np.vstack([np.full((1, 5), np.nan), 100.0 * np.diff(y, axis=0)])
    rows, windows = [], []
    audit = dict(axis_rows=len(dates), axis_head_unavailable=min(21, len(dates)),
                 axis_tail_unavailable=min(4, max(0, len(dates)-21)),
                 candidates=max(0, len(dates)-25), invalid_history_only=0,
                 invalid_future_only=0, invalid_both=0, invalid_squared_target=0)
    for t in range(21, len(dates)-4):
        hist = delta[t-20:t]
        future = delta[t:t+5, 3]
        hist_ok = bool(np.isfinite(hist).all())
        future_ok = bool(np.isfinite(future).all())
        if not hist_ok or not future_ok:
            if not hist_ok and not future_ok: audit["invalid_both"] += 1
            elif not hist_ok: audit["invalid_history_only"] += 1
            else: audit["invalid_future_only"] += 1
            continue
        with np.errstate(over="ignore"):
            q = float(np.square(future).sum())
        if not np.isfinite(q):
            audit["invalid_squared_target"] += 1
            continue
        rows.append(dict(index=t, date=dates[t], information_end=dates[t-1],
                         label_end=dates[t+4], q5=q,
                         calendar_days=int((dates[t+4]-dates[t-1]).days)))
        windows.append(hist.copy())
    d = pd.DataFrame(rows, columns=["index","date","information_end","label_end","q5","calendar_days"])
    d.index = pd.RangeIndex(len(d))
    x = np.stack(windows) if windows else np.empty((0, 20, 5), dtype=np.float64)
    audit["eligible"] = len(d)
    assert audit["candidates"] == len(d) + sum(audit[k] for k in
        ("invalid_history_only","invalid_future_only","invalid_both","invalid_squared_target"))
    d.attrs["flow"] = audit
    if len(d):
        assert (d["information_end"] < d["date"]).all()
        assert (d["date"] <= d["label_end"]).all()
        assert np.isfinite(d["q5"]).all() and (d["q5"] >= 0).all()
    return d, x, delta

def load_data(source):
    """Read six selected text columns, date-filter, then convert five numerical fields."""
    source = Path(source)
    if EXPECTED_SOURCE_SHA256 is None:
        raise RuntimeError("Historical source authorization is local only.")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Source hash mismatch: {digest}")
    text = pd.read_csv(source, usecols=["date", *COLUMNS], dtype=str, keep_default_na=False)
    normalized = text["date"].str.strip().str.replace("/", "-", regex=False)
    dates = pd.to_datetime(normalized, errors="raise")
    keep = dates <= CUTOFF
    prefix = text.loc[keep, list(COLUMNS)]
    levels = prefix.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    d, x, delta = build_dataset(dates.loc[keep], levels)
    d.attrs["source_sha256"] = digest
    d.attrs["numeric_cutoff"] = str(CUTOFF.date())
    d.attrs["columns"] = list(COLUMNS)
    if len(delta) != 3381 or len(d) != 1851:
        raise ValueError("Frozen source sample flow mismatch.")
    return d, x, delta

def split_data(d, year):
    """Return d-row positions, never original-axis indices; no flat filtering."""
    if year not in YEARS:
        raise ValueError("Only frozen years 2022/2023/2024 allowed.")
    cap = pd.Timestamp(f"{year}-12-31") if year < 2024 else CUTOFF
    ev = np.flatnonzero((d["date"].dt.year == year) & (d["label_end"] <= cap))
    if not len(ev):
        raise ValueError("No evaluation requests.")
    anchor = d.iloc[ev[0]]["date"]
    mature = np.flatnonzero(d["label_end"] < anchor)
    if len(mature) < 126:
        raise ValueError("Insufficient mature validation requests.")
    va = mature[-126:]
    tr = np.flatnonzero(d["label_end"] < d.iloc[va[0]]["date"])
    if not len(tr):
        raise ValueError("No mature training requests.")
    assert d.iloc[tr]["label_end"].max() < d.iloc[va[0]]["date"]
    assert d.iloc[va]["label_end"].max() < d.iloc[ev[0]]["date"]
    assert len(set(tr) & set(va)) == len(set(tr) & set(ev)) == len(set(va) & set(ev)) == 0
    return tr.astype(np.int64), va.astype(np.int64), ev.astype(np.int64)

def verify_frozen_counts(d):
    for year in YEARS:
        actual = tuple(len(ids) for ids in split_data(d, year))
        if actual != EXPECTED_COUNTS[year]:
            raise ValueError(f"{year}: expected {EXPECTED_COUNTS[year]}, got {actual}")
    return True

def fit_input_transform(raw_delta, d, tr):
    """Estimate only raw-day de-duplicated training moments, no optimizer/decomposition."""
    tr = np.asarray(tr, dtype=np.int64)
    if not len(tr):
        raise ValueError("No training requests.")
    targets = d.iloc[tr]["index"].to_numpy(dtype=np.int64)
    unique = np.unique(np.concatenate([np.arange(t-20, t) for t in targets]))
    raw = np.asarray(raw_delta, dtype=np.float64)[unique]
    with np.errstate(over="ignore"):
        z = np.concatenate([raw, np.square(raw)], axis=1)
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite training input statistic.")
    mean, scale = z.mean(axis=0), z.std(axis=0, ddof=0)
    zero = scale == 0
    scale[zero] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Invalid input moments.")
    return dict(mean10=mean.tolist(), scale10=scale.tolist(), zero_std_channels=np.flatnonzero(zero).tolist(),
                unique_day_indices=unique.tolist(), unique_day_count=len(unique), ddof=0,
                fit_kind="training_original_day_statistics_no_optimizer")

def transform_x(x, state):
    raw = np.asarray(x, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (20, 5):
        raise ValueError("Expected [N,20,5] raw bp changes.")
    mean = np.asarray(state["mean10"], dtype=np.float64)
    scale = np.asarray(state["scale10"], dtype=np.float64)
    if mean.shape != (10,) or scale.shape != (10,) or not (scale > 0).all():
        raise ValueError("Invalid transform state.")
    with np.errstate(over="ignore", invalid="ignore"):
        out = (np.concatenate([raw, np.square(raw)], axis=2)-mean)/scale
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite transformed input.")
    return out
