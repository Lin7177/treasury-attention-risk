"""Read only the approved V4 prediction artifact subset for V5 exploration."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from operator import index as integer_index
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("DCC", "Prior", "Uniform", "EWMA", "Kernel", "HAR_CCC", "GARCH_CCC")
UPPER = ("00", "01", "02", "11", "12", "22")
META = ("fold", "phase", "role", "period", "index", "date", "information_end", "label_end")
DATES = ("date", "information_end", "label_end")
TARGET = tuple("y" + i for i in UPPER)
FORECAST = tuple("f" + i for i in UPPER)
COMMON_DIAGONAL = ("common_d0", "common_d1", "common_d2")
READ_COLUMNS = META + ("model", "seed") + TARGET + FORECAST + COMMON_DIAGONAL
ROLES = ("validation", "evaluation")
MARKETS = {
    "china": {
        "run": "china_matrix_001",
        "analysis": "china_analysis_001",
        "phase": "exploration",
        "cutoff": "2024-06-30",
        "expected_counts": {
            ("2022", "validation"): 126, ("2022", "evaluation"): 238,
            ("2023", "validation"): 126, ("2023", "evaluation"): 238,
            ("2024", "validation"): 126, ("2024", "evaluation"): 113,
        },
    },
    "us": {
        "run": "us_matrix_001",
        "analysis": "us_analysis_001",
        "phase": "confirmation",
        "cutoff": "2025-12-31",
        "expected_counts": {("US", "validation"): 495, ("US", "evaluation"): 1496},
    },
}
V4_FREEZE_SHA256 = None


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_subset(path: Path, roles: tuple[str, ...]) -> pd.DataFrame:
    with pd.read_csv(
        path, usecols=list(READ_COLUMNS), dtype=str, keep_default_na=False,
        chunksize=20_000, encoding="utf-8", on_bad_lines="error",
    ) as chunks:
        pieces = []
        for chunk in chunks:
            keep = chunk["role"].isin(roles) & chunk["seed"].eq("0") & chunk["model"].isin(MODELS)
            if keep.any():
                pieces.append(chunk.loc[keep])
    if not pieces:
        raise ValueError("No rows in the V4 authorized role and model subset")
    return pd.concat(pieces, ignore_index=True)


def matrix(frame: pd.DataFrame, model: str | None = None) -> np.ndarray:
    """Convert six saved upper-triangle columns to matrices in 5Y, 7Y, 10Y order."""
    if model is not None and model not in MODELS:
        raise ValueError("Model not in the authorized V4 subset")
    fields = TARGET if model is None else tuple(f"{model}_{i}" for i in UPPER)
    return _symmetric(frame.loc[:, list(fields)].to_numpy(dtype=np.float64))


def _symmetric(upper: np.ndarray) -> np.ndarray:
    maps = (0, 1, 2, 1, 3, 4, 2, 4, 5)
    return upper[:, maps].reshape((-1, 3, 3))


def _typed_subset(frame: pd.DataFrame, market: str, cutoff: str) -> pd.DataFrame:
    if frame.duplicated(["fold", "index", "model"]).any():
        raise ValueError("Duplicate V4 fold/index/model identity")
    if not frame["index"].str.fullmatch(r"[0-9]+").all():
        raise ValueError("V4 index must identify an original, nonnegative integer row")
    frame = frame.copy()
    frame["index"] = frame["index"].astype("int64")
    for key in DATES:
        values = pd.to_datetime(frame[key], format="ISO8601", errors="raise")
        if values.isna().any() or values.dt.tz is not None or not values.eq(values.dt.normalize()).all():
            raise ValueError("V4 identity dates must be finite timezone-naive calendar dates")
        frame[key] = values
    if not ((frame.information_end < frame.date) & (frame.date <= frame.label_end) &
            (frame.label_end <= pd.Timestamp(cutoff))).all():
        raise ValueError("V4 information cutoff or target maturity outside the approved range")
    numeric = list(TARGET + FORECAST + COMMON_DIAGONAL)
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype("float64")
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("V4 selected predictions must contain finite matrices and scales")
    if (frame.loc[:, list(COMMON_DIAGONAL)].to_numpy() <= 0).any():
        raise ValueError("V4 common diagonal must be strictly positive")
    if not frame.phase.eq(MARKETS[market]["phase"]).all():
        raise ValueError("V4 market phase mismatch")
    return frame


def _role_ranges(frame: pd.DataFrame, market: str) -> None:
    year = frame.date.dt.year
    label_year = frame.label_end.dt.year
    validation = frame.role.eq("validation")
    evaluation = frame.role.eq("evaluation")
    if market == "us":
        periods = {2020: "2020-21", 2021: "2020-21", 2022: "2022-23",
                   2023: "2022-23", 2024: "2024-25", 2025: "2024-25"}
        valid = validation & year.between(2018, 2019) & label_year.le(2019)
        valid &= frame.period.eq("US_preconfirmation")
        eligible = evaluation & year.between(2020, 2025) & frame.period.eq(year.map(periods))
        if not (valid | eligible).all():
            raise ValueError("V4 US roles or fixed subperiod labels disagree with the frozen split")
    else:
        fold_year = frame["fold"].astype(int)
        valid = validation & year.eq(fold_year - 1) & label_year.eq(fold_year - 1)
        eligible = evaluation & year.eq(fold_year) & label_year.eq(fold_year)
        if not ((valid | eligible) & frame.period.eq(year.astype(str))).all():
            raise ValueError("V4 China roles or year labels disagree with the frozen split")


def _merge(frame: pd.DataFrame, market: str, roles: tuple[str, ...]) -> pd.DataFrame:
    spec = MARKETS[market]
    frame = _typed_subset(frame, market, spec["cutoff"])
    _role_ranges(frame, market)
    order = ["fold", "index"]
    check_fields = list(META + TARGET + COMMON_DIAGONAL)
    reference = (
        frame.loc[frame.model.eq("DCC"), check_fields].sort_values(order).reset_index(drop=True)
    )
    if reference.duplicated(order).any() or reference.duplicated(["fold", "date"]).any():
        raise ValueError("V4 reference has nonunique index or date within fold")
    expected = {k: n for k, n in spec["expected_counts"].items() if k[1] in roles}
    if Counter(zip(reference.fold, reference.role)) != expected:
        raise ValueError("V4 common request counts disagree with frozen role evidence")
    for _, fold in reference.groupby("fold", sort=False):
        if not all(fold[name].is_monotonic_increasing and not fold[name].duplicated().any()
                   for name in (*DATES, "index")):
            raise ValueError("V4 date/label/index order differs within fold")
        before = fold.loc[fold.role.eq("validation")]
        after = fold.loc[fold.role.eq("evaluation")]
        if not before.empty and not after.empty and before.label_end.max() >= after.date.min():
            raise ValueError("V4 validation labels have not all matured before evaluation starts")
    targets = matrix(reference)
    eig = np.linalg.eigvalsh(targets)
    scale = np.maximum(1.0, np.abs(eig).max(axis=1))
    if np.any(eig[:, 0] < -1e-10 * scale):
        raise ValueError("V4 target is not positive semidefinite")
    forecast_fields = {}
    for model in MODELS:
        records = frame.loc[frame.model.eq(model)].sort_values(order).reset_index(drop=True)
        if not records.loc[:, check_fields].equals(reference):
            raise ValueError(f"V4 {model} does not share DCC keys, labels, targets or common diagonal")
        forecasts = records.loc[:, list(FORECAST)].to_numpy()
        if (forecasts[:, [0, 3, 5]] <= 0).any():
            raise ValueError(f"V4 {model} forecast diagonal is nonpositive")
        try:
            np.linalg.cholesky(_symmetric(forecasts))
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"V4 {model} forecast is not positive definite") from exc
        forecast_fields.update({f"{model}_{key}": forecasts[:, i] for i, key in enumerate(UPPER)})
    context = "development" if market == "us" else "conditioning"
    reference.insert(0, "market", market)
    reference["v5_role"] = np.where(reference.role.eq("validation"), context, "exploration")
    return pd.concat([reference, pd.DataFrame(forecast_fields)], axis=1)


def load_market(root: str | Path, market: str, *, roles: tuple[str, ...] = ROLES) -> pd.DataFrame:
    raise RuntimeError("Historical replay requires the authorized local V5 artifact loader.")


def matured_positions(frame: pd.DataFrame, row_position: int) -> np.ndarray:
    """Positions with strictly matured labels in the request's market and fold."""
    i = integer_index(row_position)
    if not 0 <= i < len(frame):
        raise IndexError("V5 request position out of range")
    request = frame.iloc[i]
    mask = (frame.market.eq(request.market) & frame.fold.eq(request.fold) &
            frame.label_end.lt(request.date))
    return np.flatnonzero(mask.to_numpy())
