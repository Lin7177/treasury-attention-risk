"""Independent frozen-prefix data and training-only PCA for Target-Centric v1.

No fitting occurs at import or in load_data/split_data. The runner must claim a
market PCA budget entry before calling fit_pca on real data. Synthetic tests do
not consume that budget. Channel order is 1Y, 3Y, 5Y, 7Y, 10Y.
"""
from pathlib import Path
import numpy as np
import pandas as pd

COLS = ['yield_1y', 'yield_3y', 'yield_5y', 'yield_7y', 'yield_10y']
TARGET = 3
EXTERNAL = [0, 1, 2, 4]
WINDOW = 20
HORIZON = 5
CUTOFF = '2024-06-30'
EXPECTED = {2022: (1084, 124, 237, 8),
            2023: (1322, 120, 237, 2),
            2024: (1557, 125, 112, 2)}


def read_prefix(source):
    """Read strings first; never numerically convert sealed-date market values."""
    frame = pd.read_csv(Path(source), usecols=['date'] + COLS, dtype=str)
    dates = pd.to_datetime(frame['date'], format='mixed', errors='raise').dt.strftime('%Y-%m-%d')
    keep = dates <= CUTOFF
    approved = frame.loc[keep, COLS]
    dates = dates.loc[keep].reset_index(drop=True)
    if dates.empty:
        raise ValueError('No approved-date rows')
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('Approved date axis must be unique and increasing; do not reorder')
    levels = approved.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    return dates, levels


def build_samples(dates, levels):
    """Preserve the original supplied axis, including missing market levels."""
    dates = pd.Series(dates, dtype=str).reset_index(drop=True)
    levels = np.asarray(levels, dtype=float)
    if levels.shape != (len(dates), 5):
        raise ValueError('Expected levels[T,5] with dates[T]')
    raw_delta = np.vstack([np.full((1, 5), np.nan), np.diff(levels, axis=0) * 100])
    rows, histories, spreads = [], [], []
    for t in range(WINDOW + 1, len(levels) - HORIZON):
        history = raw_delta[t-WINDOW:t]
        prior = levels[t-1]
        endpoints = levels[[t, t+HORIZON], TARGET]
        if not (np.isfinite(history).all() and np.isfinite(prior).all() and np.isfinite(endpoints).all()):
            continue
        change = float((endpoints[1] - endpoints[0]) * 100)
        rows.append(dict(index=t, date=dates.iloc[t], label_end=dates.iloc[t+HORIZON],
                         direction=int(np.sign(change)), change_bp=change))
        histories.append(history.copy())
        spreads.append((prior[TARGET] - prior[EXTERNAL]) * 100)
    ledger = pd.DataFrame(rows, columns=['index','date','label_end','direction','change_bp'])
    return (ledger, np.asarray(histories, dtype=float).reshape(-1, WINDOW, 5),
            np.asarray(spreads, dtype=float).reshape(-1, 4), raw_delta)


def load_data(source, *, strict=True):
    """Return (ledger, X[N,20,5], spreads[N,4], raw_delta[T,5]).

    strict=False is only for synthetic input tests. Production defaults enforce
    this specific frozen-prefix contract, not a generic arbitrary-data loader.
    """
    dates, levels = read_prefix(source)
    data = build_samples(dates, levels)
    if strict:
        if len(dates) != 3381 or dates.iloc[-1] != '2024-06-28':
            raise AssertionError('Approved original prefix changed')
        if len(data[0]) != 1850:
            raise AssertionError('Common complete-sample contract changed')
    return data


def split_data(d, year, *, strict=True):
    """Annual-forward train / checkpoint-validation / evaluation positions."""
    year = int(year)
    if year not in EXPECTED:
        raise ValueError('Only frozen years 2022, 2023, 2024 are allowed')
    if not d.index.equals(pd.RangeIndex(len(d))):
        raise ValueError('Ledger must have its original consecutive position index')
    if d['date'].duplicated().any() or not d['date'].is_monotonic_increasing:
        raise ValueError('Ledger dates must be unique and chronological')
    if not d['direction'].isin([-1, 0, 1]).all() or not (d['label_end'] > d['date']).all():
        raise ValueError('Invalid ledger direction or label endpoint')
    evaluation = np.flatnonzero(((d['date'].str[:4] == str(year)) &
        (d['date'] <= '2024-06-21') & (d['label_end'] <= f'{year}-12-31')).to_numpy())
    if len(evaluation) == 0:
        raise ValueError('No evaluation requests')
    anchor = d.loc[evaluation, 'date'].min()
    validation_all = np.flatnonzero((d['label_end'] < anchor).to_numpy())[-126:]
    if len(validation_all) != 126:
        raise ValueError('Fewer than 126 strictly matured validation requests')
    start = d.loc[validation_all, 'date'].min()
    candidates = np.flatnonzero((d['label_end'] < start).to_numpy())
    train = candidates[d.loc[candidates, 'direction'].to_numpy() != 0]
    validation = validation_all[d.loc[validation_all, 'direction'].to_numpy() != 0]
    if not len(train) or not len(validation):
        raise ValueError('No nonflat train or validation requests')
    if not (d.loc[train, 'label_end'] < start).all():
        raise AssertionError('Training labels overlap validation start')
    if not (d.loc[validation_all, 'label_end'] < anchor).all():
        raise AssertionError('Validation labels are not matured before evaluation')
    if (set(train) & set(validation_all)) or (set(train) & set(evaluation)) or (set(validation_all) & set(evaluation)):
        raise AssertionError('Split positions overlap')
    if strict:
        got = (len(train), len(validation), len(evaluation), int(d.loc[evaluation,'direction'].eq(0).sum()))
        if got != EXPECTED[year]:
            raise AssertionError(f'Frozen annual flow changed: {year}: {got}')
    return train, validation, evaluation, validation_all


def fit_pca(raw_delta, d, tr):
    """Fit external-only PCA on de-duplicated raw training-window days.

    The caller owns budget claims. This function has no output I/O and does not
    mutate inputs. Covariance uses population denominator n_unique_days; this
    leaves eigenvectors and variance ratios invariant to ddof=1 rescaling.
    """
    raw_delta = np.asarray(raw_delta, dtype=float)
    tr = np.asarray(tr, dtype=int)
    if raw_delta.ndim != 2 or raw_delta.shape[1] != 5 or tr.ndim != 1 or len(tr) == 0:
        raise ValueError('Expected raw_delta[T,5] and nonempty training positions')
    if len(np.unique(tr)) != len(tr) or tr.min() < 0 or tr.max() >= len(d):
        raise ValueError('Invalid or duplicate training positions')
    if not d.iloc[tr]['direction'].isin([-1, 1]).all():
        raise ValueError('PCA training request set must be the supervised nonflat set')
    target_days = d.iloc[tr]['index'].to_numpy(dtype=int)
    if target_days.min() < WINDOW+1 or target_days.max() >= len(raw_delta):
        raise ValueError('Training source indices are outside the original axis')
    unique = np.unique(np.concatenate([np.arange(t-WINDOW, t) for t in target_days]))
    observed = raw_delta[unique][:, EXTERNAL]
    if not np.isfinite(observed).all():
        raise ValueError('Nonfinite external change in training PCA day set')
    mean = observed.mean(axis=0)
    scale = observed.std(axis=0)
    scale[scale == 0] = 1.
    z = (observed - mean) / scale
    covariance = z.T @ z / len(z)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.)
    eigenvectors = eigenvectors[:, order]
    for j in range(4):
        pivot = int(np.argmax(np.abs(eigenvectors[:, j])))
        if eigenvectors[pivot, j] < 0:
            eigenvectors[:, j] *= -1
    total = float(eigenvalues.sum())
    ratios = eigenvalues/total if total > 0 else np.zeros(4)
    return dict(mean=mean.tolist(), scale=scale.tolist(), loadings=eigenvectors[:, :3].tolist(),
        explained_variance_ratio=ratios[:3].tolist(), unique_day_indices=unique.tolist(),
        eigenvalues=eigenvalues.tolist(), all_explained_variance_ratio=ratios.tolist(),
        covariance_denominator='n_unique_train_window_days', external_channels=EXTERNAL,
        zero_total_variance=bool(total == 0))


def transform_pca(X, pca):
    X = np.asarray(X, dtype=float)
    if X.ndim != 3 or X.shape[1:] != (WINDOW, 5):
        raise ValueError('Expected X[N,20,5]')
    mean, scale, loadings = (np.asarray(pca[key], dtype=float) for key in ['mean','scale','loadings'])
    if mean.shape != (4,) or scale.shape != (4,) or loadings.shape != (4, 3):
        raise ValueError('Invalid fixed-three-component PCA shapes')
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not np.isfinite(loadings).all() or (scale <= 0).any():
        raise ValueError('Invalid PCA state')
    return ((X[:, :, EXTERNAL] - mean) / scale) @ loadings


def features(name, X, spreads, pca=None):
    X = np.asarray(X, dtype=float)
    spreads = np.asarray(spreads, dtype=float)
    if X.ndim != 3 or X.shape[1:] != (WINDOW, 5) or spreads.shape != (len(X), 4):
        raise ValueError('Expected X[N,20,5] and spreads[N,4]')
    kind = str(name).split('_', 1)[0]
    own = X[:, :, TARGET]
    if kind == 'F0':
        return own.copy()
    if kind == 'F1':
        external = X[:, :, EXTERNAL].transpose(0, 2, 1).reshape(len(X), 80)
    elif kind == 'F2':
        if pca is None:
            raise ValueError('F2 requires already-fitted, budget-claimed PCA state')
        external = transform_pca(X, pca).transpose(0, 2, 1).reshape(len(X), 60)
    else:
        raise ValueError(f'Unknown fixed linear feature recipe: {name}')
    return np.concatenate([own, external, spreads], axis=1)
