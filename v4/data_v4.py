"""V4 three-target raw-axis data contract. No network, optimizers, or old imports."""
from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

COLUMNS = ("yield_1y","yield_3y","yield_5y","yield_7y","yield_10y")
TARGET_POSITIONS = (2,3,4)
TARGET_COLUMNS = tuple(COLUMNS[i] for i in TARGET_POSITIONS)
EXPECTED_SOURCE_SHA256 = None
CUTOFF = pd.Timestamp("2024-06-30")
YEARS = (2022,2023,2024)
EXPECTED_LOCAL_COUNTS = {2022:(1105,126,238),2023:(1347,126,238),2024:(1589,126,113)}

def file_sha256(source):
    digest=hashlib.sha256()
    with Path(source).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()

def _parse_text_prefix(text,date_column,columns,cutoff,date_format):
    """Only numerical-convert the selected text prefix, never later values."""
    cutoff=pd.Timestamp(cutoff).normalize()
    if cutoff is pd.NaT or pd.isna(cutoff):
        raise ValueError("Invalid date cutoff.")
    if any(c not in text.columns for c in [date_column,*columns]):
        raise ValueError("Required date/yield column missing.")
    date_text=text[date_column].astype(str).str.strip().str.slice(0,10)
    if date_format=="%Y-%m-%d":
        date_text=date_text.str.replace("/","-",regex=False)
    dates=pd.to_datetime(date_text,format=date_format,errors="raise")
    keep=dates<=cutoff
    prefix=text.loc[keep,list(columns)]
    levels=prefix.apply(pd.to_numeric,errors="coerce").to_numpy(dtype=np.float64)
    axis_dates=pd.DatetimeIndex(dates.loc[keep])
    if axis_dates.hasnans or axis_dates.has_duplicates or not axis_dates.is_monotonic_increasing:
        raise ValueError("Original prefix dates must be unique and increasing.")
    metadata=dict(cutoff=str(cutoff.date()),axis_rows=len(axis_dates),selected_columns=list(columns),
                  cutoff_before_numeric_conversion=True,invalid_numeric_cells=int((~np.isfinite(levels)).sum()))
    return axis_dates,levels,metadata

def parse_us_text_frame(text,cutoff):
    """Synthetic parser template only; no download or authorization of real US data."""
    columns=("1 Yr","3 Yr","5 Yr","7 Yr","10 Yr")
    dates,levels,metadata=_parse_text_prefix(text,"Date",columns,cutoff,"%m/%d/%Y")
    metadata["template_scope"]="Supplied synthetic frame; real external data use requires separate authorization."
    return dates,levels,metadata

def build_dataset(dates,levels,*,cutoff=CUTOFF):
    """Construct X and future S without compressing the supplied original axis."""
    dates=pd.DatetimeIndex(pd.to_datetime(dates))
    y=np.asarray(levels,dtype=np.float64)
    if y.shape!=(len(dates),5):
        raise ValueError("Expected original levels [T,5].")
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be unique, increasing, and valid.")
    if len(dates) and dates[-1]>pd.Timestamp(cutoff):
        raise ValueError("Truncate text dates before numerical parsing.")
    with np.errstate(over="ignore",invalid="ignore"):
        delta=np.vstack([np.full((1,5),np.nan),100*np.diff(y,axis=0)])
    rows=[];windows=[];targets=[]
    flow=dict(axis_rows=len(dates),axis_head_unavailable=min(21,len(dates)),
              axis_tail_unavailable=min(4,max(0,len(dates)-21)),
              candidates=max(0,len(dates)-25),invalid_history_only=0,invalid_target_only=0,
              invalid_both=0,invalid_target_product=0)
    for t in range(21,len(dates)-4):
        past=delta[t-20:t]
        future=delta[t:t+5][:,TARGET_POSITIONS]
        history_ok=bool(np.isfinite(past).all())
        target_ok=bool(np.isfinite(future).all())
        if not history_ok or not target_ok:
            key="invalid_both" if not history_ok and not target_ok else ("invalid_history_only" if not history_ok else "invalid_target_only")
            flow[key]+=1
            continue
        with np.errstate(over="ignore",invalid="ignore"):
            target=future.T@future
        if not np.isfinite(target).all():
            flow["invalid_target_product"]+=1
            continue
        rows.append(dict(index=t,date=dates[t],information_end=dates[t-1],label_end=dates[t+4],
                         calendar_days=int((dates[t+4]-dates[t-1]).days)))
        windows.append(past.copy());targets.append(target)
    d=pd.DataFrame(rows,columns=["index","date","information_end","label_end","calendar_days"])
    d.index=pd.RangeIndex(len(d))
    rawx=np.stack(windows) if windows else np.empty((0,20,5),dtype=np.float64)
    target=np.stack(targets) if targets else np.empty((0,3,3),dtype=np.float64)
    flow["eligible"]=len(d)
    assert flow["candidates"]==len(d)+sum(flow[k] for k in
        ["invalid_history_only","invalid_target_only","invalid_both","invalid_target_product"])
    d.attrs.update(flow=flow,input_columns=list(COLUMNS),target_columns=list(TARGET_COLUMNS),
                   target_positions=list(TARGET_POSITIONS),cutoff=str(pd.Timestamp(cutoff).date()))
    if len(d):
        assert (d.information_end<d.date).all() and (d.date<=d.label_end).all()
        assert np.isfinite(rawx).all() and np.isfinite(target).all()
        assert np.allclose(target,target.swapaxes(1,2),rtol=0,atol=1e-12)
        assert (np.diagonal(target,axis1=1,axis2=2)>=0).all()
    return d,rawx,target,dates,delta

def load_data(source):
    """Authorized Chinese snapshot only, with a streaming whole-file identity hash."""
    source=Path(source)
    if EXPECTED_SOURCE_SHA256 is None:
        raise RuntimeError("Historical source authorization is local only.")
    digest=file_sha256(source)
    if digest!=EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Unexpected source SHA256: {digest}")
    text=pd.read_csv(source,usecols=["date",*COLUMNS],dtype=str,keep_default_na=False)
    dates,levels,metadata=_parse_text_prefix(text,"date",COLUMNS,CUTOFF,"%Y-%m-%d")
    d,rawx,targets,axis_dates,delta=build_dataset(dates,levels)
    if len(axis_dates)!=3381:
        raise ValueError("Authorized original-axis length changed.")
    d.attrs.update(source_sha256=digest,parse_metadata=metadata)
    return d,rawx,targets,axis_dates,delta

def split_data(d,year):
    """All returned arrays are d-row positions, never original-axis indices."""
    if year not in YEARS:
        raise ValueError("Only the approved local evaluation years are implemented.")
    cap=pd.Timestamp(f"{year}-12-31") if year<2024 else CUTOFF
    ev=np.flatnonzero((d.date.dt.year==year)&(d.label_end<=cap))
    if not len(ev):
        raise ValueError("No eligible evaluation requests.")
    mature=np.flatnonzero(d.label_end<d.iloc[ev[0]].date)
    if len(mature)<126:
        raise ValueError("Fewer than 126 mature validation requests.")
    va=mature[-126:]
    tr=np.flatnonzero(d.label_end<d.iloc[va[0]].date)
    if not len(tr):
        raise ValueError("No mature training requests.")
    assert d.iloc[tr].label_end.max()<d.iloc[va[0]].date
    assert d.iloc[va].label_end.max()<d.iloc[ev[0]].date
    assert not(set(tr)&set(va) or set(tr)&set(ev) or set(va)&set(ev))
    return tr.astype(np.int64),va.astype(np.int64),ev.astype(np.int64)

def verify_local_counts(d):
    if len(d)!=1836:
        raise ValueError(f"Three-target eligibility changed: {len(d)}.")
    for year,expected in EXPECTED_LOCAL_COUNTS.items():
        actual=tuple(map(len,split_data(d,year)))
        if actual!=expected:
            raise ValueError(f"{year}: expected {expected}, actual {actual}.")
    return True

def training_axis_indices(d,tr,axis_length=None):
    """Full raw days from earliest training input to last training label maturity."""
    tr=np.asarray(tr,dtype=np.int64)
    if not len(tr):
        raise ValueError("No training positions.")
    t=d.iloc[tr]["index"].to_numpy(dtype=np.int64)
    indices=np.arange(int(t.min())-20,int(t.max())+5,dtype=np.int64)
    if indices[0]<1 or (axis_length is not None and indices[-1]>=axis_length):
        raise ValueError("Raw training range exceeds original axis.")
    return indices

def fit_training_statistics(raw_delta,d,tr):
    """Data-dependent training statistics; no optimizer, PCA, or centering."""
    tr=np.asarray(tr,dtype=np.int64)
    if not len(tr):
        raise ValueError("Empty training request set.")
    original=d.iloc[tr]["index"].to_numpy(dtype=np.int64)
    unique=np.unique(np.concatenate([np.arange(t-20,t) for t in original]))
    raw=np.asarray(raw_delta,dtype=np.float64)
    if raw.ndim!=2 or raw.shape[1]!=5 or unique[0]<0 or unique[-1]>=len(raw):
        raise ValueError("Invalid original-day shape/range.")
    values=raw[unique]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite required training input day.")
    with np.errstate(over="ignore",invalid="ignore"):
        rms=np.sqrt(np.square(values).mean(axis=0))
    zero=np.flatnonzero(rms<=0)
    if len(zero) or not np.isfinite(rms).all():
        raise ValueError(f"Invalid/zero training RMS channels: {zero.tolist()}")
    normalized=values/rms
    target=normalized[:,TARGET_POSITIONS]
    ctrain=target.T@target/len(target)
    c0=.99*ctrain+.01*np.eye(3)
    if not np.isfinite(c0).all():
        raise ValueError("Nonfinite training prior.")
    return dict(rms5=rms.tolist(),Ctrain=ctrain.tolist(),C0=c0.tolist(),
                unique_day_indices=unique.tolist(),unique_day_count=len(unique),
                training_requests=len(tr),zero_rms_channels=[],invalid_training_cells=0,
                target_positions=list(TARGET_POSITIONS),centered=False,
                prior_units="one_day_normalized_outer_product",
                statistics_are_data_learning=True,optimizer_fits=0,decomposition_fits=0)

def _rms(state):
    rms=np.asarray(state["rms5"],dtype=np.float64)
    if rms.shape!=(5,) or not np.isfinite(rms).all() or not(rms>0).all():
        raise ValueError("Invalid five-channel RMS state.")
    return rms

def normalize_inputs(rawx,state):
    x=np.asarray(rawx,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(20,5):
        raise ValueError("Expected rawx[N,20,5].")
    z=x/_rms(state)
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite normalized input.")
    return z

def normalize_targets(targets,state):
    a=np.asarray(targets,dtype=np.float64)
    if a.ndim!=3 or a.shape[1:]!=(3,3):
        raise ValueError("Expected targets[N,3,3].")
    rms=_rms(state)[list(TARGET_POSITIONS)]
    out=a/np.outer(rms,rms)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite normalized target.")
    return out
