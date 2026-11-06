"""Unique pre-agreed matured-label prior side diagnostic; zero optimizer/decomposition.

This program is intentionally separate from Q5. Run only after V3 protocol freeze.
It reads legacy local artifacts, never imports legacy runners or reads sealed data.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from data_v3 import CUTOFF

WINDOW = 126
PSEUDOCOUNT = 20
PROBABILITY_CLIP = 1e-15

def select_saved_models(saved):
    """Exact frozen legacy identifiers only; no fuzzy/prefix model selection."""
    mapping = {"N0_OwnMaskedFixed": "N0", "N3_ContentCross": "N3"}
    selected = saved.loc[saved["model"].isin(mapping)].copy()
    selected["model"] = selected["model"].map(mapping)
    return selected


def prior_states(ledger, requests, train_prior):
    """Causal state: only direction values whose label_end < request date."""
    history = ledger.sort_values(["label_end","date","index"]).reset_index(drop=True)
    rows = []
    previous_count = None
    previous_q = None
    previous_window = None
    for request in requests.sort_values(["date","index"]).itertuples(index=False):
        mature = history.loc[history["label_end"] < request.date]
        mature = mature.loc[mature["direction"] != 0]
        used = mature.tail(WINDOW)
        n = len(used)
        up = int((used["direction"] == 1).sum())
        q = (up + PSEUDOCOUNT*train_prior)/(n+PSEUDOCOUNT)
        if n:
            assert used["label_end"].max() < request.date
        ids = tuple(used["index"].astype(int))
        rows.append(dict(index=int(request.index),date=request.date,
                         probability_up=float(q),history_count=n,
                         total_matured_nonflat=len(mature),history_up=up,
                         max_mature_label_end=used["label_end"].max() if n else pd.NaT,
                         new_matured_since_previous=None if previous_count is None else len(mature)-previous_count,
                         q_changed_since_previous=None if previous_q is None else bool(q != previous_q),
                         window_changed_since_previous=None if previous_window is None else bool(ids != previous_window)))
        previous_count, previous_q, previous_window = len(mature), q, ids
    return pd.DataFrame(rows)

def metrics(direction, probability):
    y, p = np.asarray(direction), np.asarray(probability,dtype=float)
    if not np.isfinite(p).all() or ((p < 0)|(p > 1)).any():
        raise ValueError("Illegal saved probability.")
    pred = np.where(p >= .5,1,-1)
    nonflat = y != 0
    yu, pu = (y[nonflat]==1).astype(float), p[nonflat]
    up = y == 1; down = y == -1
    recalls=[float((pred[m]==y[m]).mean()) if m.any() else np.nan for m in (up,down)]
    pc=np.clip(pu,PROBABILITY_CLIP,1-PROBABILITY_CLIP)
    return dict(n_all=len(y),n_nonflat=int(nonflat.sum()),n_flat=int((~nonflat).sum()),
                accuracy_all=float((pred==y).mean()) if len(y) else None,
                balanced_accuracy_nonflat=float(np.mean(recalls)) if all(np.isfinite(recalls)) else None,
                brier_nonflat=float(np.mean((pu-yu)**2)) if len(yu) else None,
                logloss_nonflat=float(-np.mean(yu*np.log(pc)+(1-yu)*np.log1p(-pc))) if len(yu) else None,
                predicted_up_fraction=float((pred==1).mean()) if len(y) else None)

def run(legacy_dir, output):
    legacy_dir, output = Path(legacy_dir),Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite diagnostic output.")
    ledger_path=legacy_dir/"sample_ledger.csv"
    predictions_path=legacy_dir/"predictions_ensemble.csv"
    manifest_path=legacy_dir/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    raise RuntimeError("Historical source authorization is local only.")
    ledger=pd.read_csv(ledger_path)
    for col in ("date","label_end"):
        ledger[col]=pd.to_datetime(ledger[col])
    if len(ledger)!=1850 or ledger["index"].duplicated().any() or ledger["date"].duplicated().any():
        raise ValueError("Legacy common ledger key/count mismatch.")
    if (ledger["label_end"]>CUTOFF).any() or not ledger["direction"].isin([-1,0,1]).all():
        raise ValueError("Unapproved legacy labels.")
    if not np.array_equal(np.sign(ledger["change_bp"].to_numpy()),ledger["direction"].to_numpy()):
        raise ValueError("Legacy label/direction mismatch.")
    saved=pd.read_csv(predictions_path)
    saved["date"]=pd.to_datetime(saved["date"])
    wanted=select_saved_models(saved)
    if wanted.duplicated(["model","index","date"]).any():
        raise ValueError("Duplicate legacy prediction key.")
    if (wanted["date"]>CUTOFF).any():
        raise ValueError("Unapproved legacy prediction date.")
    all_states=[]; all_predictions=[]
    counts={2022:237,2023:237,2024:112}
    for year in (2022,2023,2024):
        original=wanted.loc[(wanted["year"]==year)&(wanted["model"]=="N0")].sort_values("date")
        if len(original)!=counts[year]:
            raise ValueError("Legacy evaluation count mismatch.")
        request=ledger.merge(original[["index","date"]],on=["index","date"],validate="one_to_one").sort_values("date")
        if len(request)!=counts[year]:
            raise ValueError("Legacy key mismatch.")
        anchor=request["date"].min()
        validation=ledger.loc[ledger["label_end"]<anchor].tail(126)
        training=ledger.loc[(ledger["label_end"]<validation["date"].min())&(ledger["direction"]!=0)]
        if len(training)!={2022:1084,2023:1322,2024:1557}[year]:
            raise ValueError("Legacy mature training count mismatch.")
        pi=float((training["direction"]==1).mean())
        state=prior_states(ledger,request,pi)
        state["year"]=year;state["train_prior"]=pi
        all_states.append(state)
        for model in ("N0","N3"):
            sub=wanted.loc[(wanted["year"]==year)&(wanted["model"]==model)]
            pair=request[["index","date","direction"]].merge(sub[["index","date","probability_up"]],on=["index","date"],validate="one_to_one")
            if len(pair)!=len(request): raise ValueError("Incomplete model pairing.")
            pair["model"]=model;pair["year"]=year
            all_predictions.append(pair)
        for model,probs in (("Q126_20",state["probability_up"].to_numpy()),("TRAIN_PRIOR",np.full(len(request),pi))):
            row=request[["index","date","direction"]].copy()
            row["probability_up"]=probs;row["model"]=model;row["year"]=year
            all_predictions.append(row)
    states=pd.concat(all_states,ignore_index=True)
    prediction=pd.concat(all_predictions,ignore_index=True)
    if len(states)!=586: raise ValueError("Expected 586 prior request reconstructions.")
    output.mkdir(parents=True)
    states.to_csv(output/"prior_states.csv",index=False)
    prediction.to_csv(output/"paired_predictions.csv",index=False)
    results=[]
    for (model,year),g in prediction.groupby(["model","year"]):
        results.append(dict(model=model,year=int(year),**metrics(g["direction"],g["probability_up"])))
    for model,g in prediction.groupby("model"):
        results.append(dict(model=model,year="POOLED",**metrics(g["direction"],g["probability_up"])))
        per=[r for r in results if r["model"]==model and isinstance(r["year"],int)]
        results.append(dict(model=model,year="MACRO",
            **{key:float(np.mean([r[key] for r in per])) for key in
                ("accuracy_all","balanced_accuracy_nonflat","brier_nonflat","logloss_nonflat","predicted_up_fraction")}))
    pd.DataFrame(results).to_csv(output/"metrics.csv",index=False)
    report=dict(kind="exploratory_matured_label_feedback_side_diagnostic",optimizer_fits=0,
        request_reconstructions=586,window=WINDOW,pseudocount=PSEUDOCOUNT,
        newly_matured_labels_after_first_request_per_year=int(states["new_matured_since_previous"].dropna().sum()),
        requests_with_new_matured_labels_after_first_per_year=int((states["new_matured_since_previous"].fillna(0)>0).sum()),
        q_value_changes_within_year=int(states["q_changed_since_previous"].fillna(False).astype(bool).sum()),
        window_changes_within_year=int(states["window_changed_since_previous"].fillna(False).astype(bool).sum()),
        initial_state_reconstructions=3,probability_clip_for_logloss=PROBABILITY_CLIP,
        feedback_information_difference="q uses matured in-year labels; frozen N0/N3 do not update with those labels",
        limitations=["legacy exposed direction task; distinct from Q5 589 samples","no window comparison, offsets or refits","proper scores exclude flat; all-request accuracy retains flat as error","macro descriptive metrics are year-equal; no uncertainty intervals claimed here"],
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ledger_path,predictions_path,manifest_path)})
    (output/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--legacy-dir",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.legacy_dir,args.output),ensure_ascii=False,indent=2))
