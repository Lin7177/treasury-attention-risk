"""V4 fixed matrix-risk analysis. No model fitting or market downloads."""
from __future__ import annotations
import os
for _thread_variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable]="1"
import argparse,hashlib,json,platform,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from clock_v4 import ScientificClock

ROOT=Path(__file__).resolve().parent
MODELS=("Prior","Uniform","EWMA","Kernel","HAR_CCC","GARCH_CCC","DCC","Static","MLP","Content")
LEARNED=("Static","MLP","Content")
BASELINES=("Static","MLP","Kernel","EWMA","HAR_CCC","GARCH_CCC","DCC")
FINANCIAL=("HAR_CCC","GARCH_CCC","DCC")
SEEDS=(2026110541,2026110542,2026110543)
TRI=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
YCOL=["y00","y01","y02","y11","y12","y22"]
FCOL=[c.replace("y","f",1) for c in YCOL]
DCOL=["common_d0","common_d1","common_d2"]
KEY=["phase","fold","role","index","date"]
META=["information_end","label_end","period"]+YCOL+DCOL
PROJECTIONS={"own7":np.array([0.,1.,0.]),"slope10_5":np.array([-1.,0.,1.]),
             "fly7_5_10":np.array([-.6,1.,-.4])}
PHASE_PERIODS={"exploration":("2022","2023","2024"),
               "confirmation":("2020-21","2022-23","2024-25")}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def matrices(frame,columns):
    a=frame[list(columns)].to_numpy(dtype=float);out=np.zeros((len(a),3,3))
    for k,(i,j) in enumerate(TRI):out[:,i,j]=a[:,k];out[:,j,i]=a[:,k]
    return out

def validate_matrix(a,positive=False):
    a=np.asarray(a,float)
    if a.ndim!=3 or a.shape[1:]!=(3,3) or not len(a) or not np.isfinite(a).all():
        raise ValueError("Finite nonempty [N,3,3] required")
    if not np.allclose(a,a.swapaxes(1,2),rtol=1e-10,atol=1e-10):
        raise ValueError("Nonsymmetric matrix")
    if positive:np.linalg.cholesky(a)
    else:
        tol=1e-10*np.maximum(1,np.max(np.abs(a),axis=(1,2)))
        if (np.linalg.eigvalsh(a)[:,0]<-tol).any():raise ValueError("NonPSD target")
    return a

def matrix_loss(s,h):
    s=validate_matrix(s);h=validate_matrix(h,True)
    if s.shape!=h.shape:raise ValueError("Target/forecast shape mismatch")
    c=np.linalg.cholesky(h)
    result=(2*np.log(np.diagonal(c,axis1=1,axis2=2)).sum(1)+np.trace(np.linalg.solve(h,s),axis1=1,axis2=2))/3
    if not np.isfinite(result).all():raise ValueError("Nonfinite matrix loss")
    return result

def common_matrix(h,d):
    h=validate_matrix(h,True);d=np.asarray(d,float)
    if d.shape!=(len(h),3) or not np.isfinite(d).all() or (d<=0).any():
        raise ValueError("Positive aligned physical common diagonal required")
    roots=np.sqrt(np.diagonal(h,axis1=1,axis2=2))
    r=h/roots[:,:,None]/roots[:,None,:]
    out=r*np.sqrt(d)[:,:,None]*np.sqrt(d)[:,None,:]
    return validate_matrix(out,True)

def quadratic(s,a):
    raw=np.einsum("ni,nij,nj->n",a,s,a)
    tol=1e-10*np.maximum(1,np.max(np.abs(s),axis=(1,2)))*np.square(np.abs(a).sum(1))
    if not np.isfinite(raw).all() or (raw < -tol).any():raise ValueError("Illegal quadratic risk")
    hits=int((raw<0).sum())
    return np.maximum(raw,0),hits

def hedge_risk(s,h):
    s=validate_matrix(s);h=validate_matrix(h,True);n=len(s)
    neighbor=h[:,[0,2]][:,:,[0,2]];cross=h[:,[0,2],1]
    ridge=.01*np.trace(neighbor,axis1=1,axis2=2)/2
    beta=np.linalg.solve(neighbor+ridge[:,None,None]*np.eye(2),cross[...,None]).squeeze(-1)
    norm=np.abs(beta).sum(1);hit=norm>2
    beta[hit]*=(2/norm[hit])[:,None]
    a=np.column_stack([-beta[:,0],np.ones(n),-beta[:,1]])
    risk,roundoff=quadratic(s,a)
    return risk,beta,dict(cap_hits=int(hit.sum()),quadratic_roundoff_hits=roundoff,
                          ridge_min=float(ridge.min()),ridge_max=float(ridge.max()))

def scalar_loss(q,f):
    if not np.isfinite(q).all() or not np.isfinite(f).all() or (q<0).any() or (f<=0).any():
        raise ValueError("Invalid projected risk target/forecast")
    out=np.log(f)+q/f
    if not np.isfinite(out).all():raise ValueError("Nonfinite projected loss")
    return out

def prepare(frame):
    f=frame.copy()
    required=set(KEY+META+["model","seed"]+FCOL)
    if required-set(f):raise ValueError("Missing columns "+str(sorted(required-set(f))))
    for col in ["phase","fold","role","model","period"]:f[col]=f[col].astype(str)
    if not set(f.phase)<=set(PHASE_PERIODS) or len(set(f.phase))!=1:raise ValueError("One recognized phase per analysis")
    if not set(f.role)<={"train","validation","evaluation"}:raise ValueError("Unknown role")
    for col in YCOL+FCOL+DCOL+["index","seed"]:
        f[col]=pd.to_numeric(f[col],errors="raise")
        if not np.isfinite(f[col]).all():raise ValueError("Nonfinite column "+col)
    for col in ["index","seed"]:
        if not np.equal(f[col],np.floor(f[col])).all():raise ValueError("Noninteger key "+col)
        f[col]=f[col].astype(np.int64)
    for col in ["date","information_end","label_end"]:
        p=pd.to_datetime(f[col],errors="raise")
        if p.isna().any():raise ValueError("Missing date")
        f[col]=p.dt.strftime("%Y-%m-%d")
    if (f.information_end>=f.date).any() or (f.label_end<f.date).any():raise ValueError("Invalid information/maturity order")
    if (f[DCOL]<=0).any().any():raise ValueError("Nonpositive common diagonal")
    if f.duplicated(KEY+["model","seed"]).any():raise ValueError("Duplicate seed/request key")
    validate_matrix(matrices(f,YCOL));validate_matrix(matrices(f,FCOL),True)
    return f

def ensemble(frame,enforce_counts=True):
    f=prepare(frame);phase=f.phase.iloc[0]
    if set(f.model)!=set(MODELS):raise ValueError("Exactly ten registered models required")
    # Validate shared ledger and complete seeds within each existing role/fold.
    outputs=[]
    for (fold,role),part in f.groupby(["fold","role"],sort=True):
        names=set(part.model)
        if role=="evaluation" and names!=set(MODELS):raise ValueError("Incomplete evaluation models")
        reference=None
        for model in MODELS:
            rows=part[part.model==model]
            if not len(rows):continue
            seeds=set(SEEDS) if model in LEARNED else {0}
            if set(rows.seed)!=seeds:raise ValueError("Wrong seeds "+model)
            predictions=[];ownref=None
            for seed in sorted(seeds):
                item=rows[rows.seed==seed].sort_values(KEY).reset_index(drop=True)
                if item.duplicated(["index"]).any() or item.duplicated(["date"]).any():raise ValueError("Index/date not one-to-one")
                if ownref is None:ownref=item
                elif not item[KEY+META].equals(ownref[KEY+META]):raise ValueError("Mismatched seed ledger")
                if reference is None:reference=item[KEY+META]
                elif not item[KEY+META].equals(reference):raise ValueError("Mismatched model ledger/target/commonD")
                predictions.append(item[FCOL].to_numpy())
            result=ownref[KEY+META].copy();result["model"]=model
            # Physical SPD matrices first; correlations/hedges are derived later.
            result[FCOL]=np.stack(predictions).mean(0)
            outputs.append(result)
    e=pd.concat(outputs,ignore_index=True)
    ev=e[e.role=="evaluation"]
    ref=ev[ev.model=="Prior"].sort_values(KEY).reset_index(drop=True)
    if not len(ref) or ref.date.duplicated().any() or ref["index"].duplicated().any():raise ValueError("Missing/duplicate evaluation reference")
    if not ref.date.is_monotonic_increasing:raise ValueError("Evaluation chronology")
    periods=PHASE_PERIODS[phase]
    if set(ref.period)!=set(periods):raise ValueError("Expected three fixed periods")
    years=pd.to_datetime(ref.date).dt.year
    if phase=="exploration":
        if set(ref.fold)!={"2022","2023","2024"} or not (ref.fold==years.astype(str)).all():raise ValueError("China fold/year mismatch")
        if not (ref.period==years.astype(str)).all():raise ValueError("China period/year mismatch")
        if (ref.label_end>"2024-06-30").any() or (pd.to_datetime(ref.label_end).dt.year!=years).any():raise ValueError("China end boundary")
        if enforce_counts and ref.groupby("period").size().to_dict()!={"2022":238,"2023":238,"2024":113}:raise ValueError("Expected 589 China requests")
    else:
        expected=years.map({2020:"2020-21",2021:"2020-21",2022:"2022-23",2023:"2022-23",2024:"2024-25",2025:"2024-25"})
        if set(ref.fold)!={"US"} or not (expected==ref.period).all() or (ref.label_end>"2025-12-31").any():raise ValueError("US confirmation boundary")
        if enforce_counts and (set(years)!={2020,2021,2022,2023,2024,2025} or len(ref)<500):raise ValueError("Incomplete confirmation years/coverage")
    for fold,part in e.groupby("fold"):
        role_ref={r:p[p.model==p.model.iloc[0]] for r,p in part.groupby("role")}
        if "train" in role_ref and "validation" in role_ref:
            if role_ref["train"].label_end.max()>=role_ref["validation"].date.min():raise ValueError("Train labels not mature before validation")
        if "validation" in role_ref and "evaluation" in role_ref:
            if role_ref["validation"].label_end.max()>=role_ref["evaluation"].date.min():raise ValueError("Validation labels not mature before evaluation")
    pred=[]
    for name in MODELS:
        item=ev[ev.model==name].sort_values(KEY).reset_index(drop=True)
        if not item[KEY+META].equals(ref[KEY+META]):raise ValueError("Evaluation ledger mismatch")
        pred.append(matrices(item,FCOL))
    return f,e,ref,np.stack(pred,axis=1)

def model_arrays(ref,pred):
    target=matrices(ref,YCOL);diagonal=ref[DCOL].to_numpy()
    n=len(ref);loss=np.empty((n,10));shape=np.empty_like(loss);risk=np.empty_like(loss)
    info=[]
    for j,name in enumerate(MODELS):
        loss[:,j]=matrix_loss(target,pred[:,j])
        shape[:,j]=matrix_loss(target,common_matrix(pred[:,j],diagonal))
        risk[:,j],beta,numerics=hedge_risk(target,pred[:,j])
        info.append(dict(model=name,**numerics))
    return target,loss,shape,risk,pd.DataFrame(info)

def means_by_period(v,period):
    p=np.asarray(period);labels=sorted(set(p));v=np.asarray(v,float)
    return {s:v[p==s].mean(0) for s in labels}

def risk_ratio(numerator,denominator):
    n=np.asarray(numerator,float);d=np.asarray(denominator,float)
    out=np.full(np.broadcast_shapes(n.shape,d.shape),np.nan)
    np.divide(n,d,out=out,where=d>0)
    return out-1

def statistics(full,shape,risk,period):
    p=np.asarray(period);labels=sorted(set(p));c=MODELS.index("Content")
    f=full[:,c,None]-full[:,[MODELS.index(v) for v in BASELINES]]
    s=shape[:,c,None]-shape[:,[MODELS.index(v) for v in BASELINES]]
    rm=means_by_period(risk,p)
    by={label:risk_ratio(rm[label][c],rm[label][[MODELS.index(v) for v in FINANCIAL]]) for label in labels}
    result={}
    for family,array in [("full",f),("shape",s)]:
        annual=means_by_period(array,p)
        result[family]={"macro":np.stack(list(annual.values())).mean(0),"pooled":array.mean(0),**annual}
    result["hedge"]={"macro":np.stack(list(by.values())).mean(0),
        "pooled":risk_ratio(risk[:,c].mean(),risk[:,[MODELS.index(v) for v in FINANCIAL]].mean(0)),**by}
    return result

def block_indices(n,b,rng):
    if b<1 or n<b:raise ValueError("Need period N>=block>=1")
    starts=rng.integers(0,n-b+1,size=(n+b-1)//b)
    return (starts[:,None]+np.arange(b)).reshape(-1)[:n]

def bootstrap(full,shape,risk,period,b,reps=2000,seed=2026110540,check=lambda:None):
    p=np.asarray(period);labels=sorted(set(p));groups=[np.flatnonzero(p==v) for v in labels]
    point=statistics(full,shape,risk,p)
    out={fam:{stat:np.empty((reps,len(v))) for stat,v in values.items()} for fam,values in point.items()}
    rng=np.random.default_rng(seed)
    for iteration in range(reps):
        check()
        ix=np.concatenate([ids[block_indices(len(ids),b,rng)] for ids in groups])
        got=statistics(full[ix],shape[ix],risk[ix],p[ix])
        for fam,values in got.items():
            for stat,value in values.items():out[fam][stat][iteration]=value
    return out

def interval_rows(full,shape,risk,period,cfg,check=lambda:None):
    point=statistics(full,shape,risk,period);rows=[];inf=cfg["inference"]
    for b in inf["blocks"]:
        draw=bootstrap(full,shape,risk,period,b,inf["replicates"],inf["seed"],check)
        for family,stats in point.items():
            baselines=FINANCIAL if family=="hedge" else BASELINES
            upper_p=1-.05/len(baselines)
            for stat,values in stats.items():
                for j,base in enumerate(baselines):
                    v=draw[family][stat][:,j];valid=np.isfinite(v);good=bool(valid.all())
                    lo,hi=np.quantile(v,[.025,.975]) if good else (np.nan,np.nan)
                    upper=np.quantile(v,upper_p) if good else np.nan
                    rows.append(dict(family=family,model="Content",baseline=base,statistic=stat,
                        block=b,estimate=values[j],lower_95=lo,upper_95=hi,
                        family_upper=upper,family_upper_quantile=upper_p,
                        replications=len(v),valid_replications=int(valid.sum()),
                        missing_reason="" if good else "undefined_ratio_in_resample_no_refill"))
    return pd.DataFrame(rows)

def aggregate_metrics(ref,pred):
    target,full,shape,risk,numerics=model_arrays(ref,pred)
    p=ref.period.to_numpy();yr=pd.to_datetime(ref.date).dt.year.to_numpy()
    rows=[];projections=[]
    for j,name in enumerate(MODELS):
        diff=pred[:,j]-target
        per={"full_qlike":full[:,j],"common_d_qlike":shape[:,j],"hedge_risk":risk[:,j],
             "frobenius_mse_per_entry":np.square(diff).sum((1,2))/9,
             "mean_frobenius_norm":np.linalg.norm(diff,axis=(1,2)),
             "forecast_trace":np.trace(pred[:,j],axis1=1,axis2=2),
             "actual_trace":np.trace(target,axis1=1,axis2=2)}
        for k,label in enumerate(("5y","7y","10y")):
            per["diagonal_mse_"+label]=np.square(diff[:,k,k])
            per["mean_forecast_"+label]=pred[:,j,k,k]
            per["mean_actual_"+label]=target[:,k,k]
        byperiod={label:{k:float(v[p==label].mean()) for k,v in per.items()} for label in sorted(set(p))}
        for label,values in byperiod.items():rows.append(dict(model=name,aggregation="period",group=label,n=int((p==label).sum()),**values))
        for year in sorted(set(yr)):rows.append(dict(model=name,aggregation="year",group=str(year),n=int((yr==year).sum()),**{k:float(v[yr==year].mean()) for k,v in per.items()}))
        rows.append(dict(model=name,aggregation="pooled",group="all",n=len(p),**{k:float(v.mean()) for k,v in per.items()}))
        rows.append(dict(model=name,aggregation="macro",group="all",n=len(p),**{k:float(np.mean([v[k] for v in byperiod.values()])) for k in per}))
        for projection,a in PROJECTIONS.items():
            aa=np.broadcast_to(a,(len(p),3));q,qt=quadratic(target,aa);f,ft=quadratic(pred[:,j],aa)
            l=scalar_loss(q,f)
            for label in sorted(set(p)):
                ix=p==label
                projections.append(dict(model=name,projection=projection,period=label,n=int(ix.sum()),
                    actual_energy=q[ix].mean(),forecast_energy=f[ix].mean(),qlike=l[ix].mean(),
                    mse=np.square(f[ix]-q[ix]).mean(),quadratic_roundoff_hits=qt+ft))
    return pd.DataFrame(rows),pd.DataFrame(projections),numerics,(target,full,shape,risk)

def tail_table(ref,full,shape,risk):
    p=ref.period.to_numpy();target=matrices(ref,YCOL);trace=np.trace(target,axis1=1,axis2=2);rows=[]
    point=statistics(full,shape,risk,p)
    for label in sorted(set(p)):
        ix=np.flatnonzero(p==label);ordered=ix[np.argsort(-trace[ix],kind="stable")]
        tail=ordered[:int(np.ceil(.01*len(ix)))];largest=ordered[:1]
        for family in ["full","shape","hedge"]:
            bases=FINANCIAL if family=="hedge" else BASELINES
            for base in bases:
                c=MODELS.index("Content");b=MODELS.index(base)
                keep=np.setdiff1d(ix,tail);keep1=np.setdiff1d(ix,largest)
                if family=="hedge":
                    dropped=risk_ratio(risk[keep,c].mean(),risk[keep,b].mean())
                    one=risk_ratio(risk[keep1,c].mean(),risk[keep1,b].mean())
                    contribution=(risk[tail,c]-risk[tail,b]).sum()/len(ix)
                else:
                    v=(full if family=="full" else shape);delta=v[:,c]-v[:,b]
                    dropped=delta[keep].mean();one=delta[keep1].mean()
                    contribution=delta[tail].sum()/len(ix)
                j=list(bases).index(base)
                rows.append(dict(family=family,baseline=base,period=label,n=len(ix),n_tail=len(tail),
                    tail_dates=";".join(ref.iloc[tail].date),trace_share=trace[tail].sum()/trace[ix].sum() if trace[ix].sum()>0 else np.nan,
                    full_estimate=point[family][label][j],without_top1pct=dropped,
                    without_max_trace=one,tail_contribution_to_mean_difference=contribution,
                    interpretation="ex_post_only_main_sample_unchanged"))
    return pd.DataFrame(rows)

def decision(intervals,point,periods,cfg):
    flags={};checks=[]
    thresholds={"full":cfg["inference"]["primary_minimum_point_delta"],"shape":0.,
                "hedge":-cfg["hedge"]["minimum_point_reduction"]}
    for family in ["full","shape","hedge"]:
        bases=FINANCIAL if family=="hedge" else BASELINES;passed=[]
        for j,base in enumerate(bases):
            row=intervals[(intervals.family==family)&(intervals.baseline==base)&(intervals.statistic=="macro")&(intervals.block==10)]
            if len(row)!=1:raise ValueError("Missing unique main block decision row")
            r=row.iloc[0];neg=sum(np.isfinite(point[family][v][j]) and point[family][v][j]<0 for v in periods)
            value=point[family]["macro"][j]
            size=value<0 if family=="shape" else value<=thresholds[family]
            ok=bool(np.isfinite(value) and size and np.isfinite(r.family_upper) and r.family_upper<0 and neg>=2)
            passed.append(ok);checks.append(dict(family=family,baseline=base,estimate=value,upper=r.family_upper,negative_periods=int(neg),passed=ok))
        flags[{"full":"full_matrix_signal","shape":"dependence_shape_signal","hedge":"hedge_proxy_signal"}[family]]=all(passed)
    return dict(**flags,overall_proxy_advantage=all(flags.values()),checks=checks,
        interpretation="Point minimum effect plus interval direction support; not confidence proof of minimum magnitude, pure correlation identification, or tradable PnL")

def validate_config(cfg):
    if tuple(cfg["models"])!=MODELS or tuple(cfg["learned"])!=LEARNED or tuple(cfg["primary_baselines"])!=BASELINES or tuple(cfg["financial_baselines"])!=FINANCIAL:
        raise ValueError("Configuration/model contracts differ")
    if tuple(cfg["seeds"])!=SEEDS or cfg["matrix_loss_divisor"]!=3:raise ValueError("Seed or loss mismatch")
    inf=cfg["inference"]
    if set(inf["blocks"])!={5,10,20} or inf["replicates"]!=2000 or inf["seed"]!=2026110540:raise ValueError("Frozen inference contract differs")
    if inf["primary_minimum_point_delta"]!=-.01 or inf["negative_subperiods_required"]!=2 or cfg["hedge"]["minimum_point_reduction"]!=.05:
        raise ValueError("Frozen point threshold differs")
    if (inf["full_family_size"],inf["shape_family_size"],inf["hedge_family_size"])!=(7,7,3):raise ValueError("Family size mismatch")
    if (cfg["hedge"]["target_coordinate"],tuple(cfg["hedge"]["neighbor_coordinates"]),cfg["hedge"]["ridge_fraction"],cfg["hedge"]["neighbor_l1_cap"])!=(1,(0,2),.01,2):
        raise ValueError("Hedge decision contract differs")
    return cfg

def seed_role_metrics(f):
    rows=[]
    for (phase,fold,role,model,seed),part in f.groupby(["phase","fold","role","model","seed"],sort=True):
        s=matrices(part,YCOL);h=matrices(part,FCOL);l=matrix_loss(s,h)
        shape=matrix_loss(s,common_matrix(h,part[DCOL].to_numpy()))
        risk,_,_=hedge_risk(s,h)
        values={"full_qlike":l,"common_d_qlike":shape,"hedge_risk":risk,
                "frobenius_mse_per_entry":np.square(h-s).sum((1,2))/9}
        period=part.period.to_numpy();year=pd.to_datetime(part.date).dt.year.to_numpy()
        summary=[]
        for label in sorted(set(period)):
            ix=period==label;summary.append({k:float(v[ix].mean()) for k,v in values.items()})
            rows.append(dict(phase=phase,fold=fold,role=role,model=model,seed=int(seed),
                aggregation="period",group=label,n=int(ix.sum()),**summary[-1]))
        for label in sorted(set(year)):
            ix=year==label
            rows.append(dict(phase=phase,fold=fold,role=role,model=model,seed=int(seed),
                aggregation="year",group=str(label),n=int(ix.sum()),
                **{k:float(v[ix].mean()) for k,v in values.items()}))
        rows.append(dict(phase=phase,fold=fold,role=role,model=model,seed=int(seed),
            aggregation="pooled",group="all",n=len(part),**{k:float(v.mean()) for k,v in values.items()}))
        rows.append(dict(phase=phase,fold=fold,role=role,model=model,seed=int(seed),
            aggregation="macro",group="all",n=len(part),**{k:float(np.mean([r[k] for r in summary])) for k in values}))
    return pd.DataFrame(rows)

def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean_json(v) for v in value]
    if isinstance(value,np.ndarray):return clean_json(value.tolist())
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.bool_,)):return bool(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value

def analyze_frames(frame,cfg,check=lambda:None,enforce_counts=True):
    validate_config(cfg);f,e,ref,pred=ensemble(frame,enforce_counts)
    phase=ref.phase.iloc[0]
    metrics,projections,numerics,(target,full,shape,risk)=aggregate_metrics(ref,pred)
    check();p=ref.period.to_numpy();point=statistics(full,shape,risk,p)
    intervals=interval_rows(full,shape,risk,p,cfg,check)
    deltas=[]
    for family,values in point.items():
        bases=FINANCIAL if family=="hedge" else BASELINES
        for statistic,array in values.items():
            for base,value in zip(bases,array):deltas.append(dict(family=family,baseline=base,statistic=statistic,estimate=value))
    fixed=[]
    for name,a in {"unhedged":np.array([0.,1.,0.]),"fixed_0_6_0_4":np.array([-.6,1.,-.4])}.items():
        r,hit=quadratic(target,np.broadcast_to(a,(len(target),3)))
        for period in sorted(set(p)):fixed.append(dict(strategy=name,period=period,n=int((p==period).sum()),mean_risk=r[p==period].mean(),roundoff_hits=hit))
    outputs=dict(model_metrics=metrics[metrics.aggregation.isin(["macro","pooled"])],
        period_metrics=metrics[metrics.aggregation=="period"],year_metrics=metrics[metrics.aggregation=="year"],
        seed_role_metrics=seed_role_metrics(f),
        seed_metrics=seed_role_metrics(f[f.role=="evaluation"].assign(fold="ALL")),
        paired_intervals=intervals,paired_point_differences=pd.DataFrame(deltas),
        projection_metrics=projections,hedge_diagnostics=numerics,fixed_exposure_risk=pd.DataFrame(fixed),
        tail_sensitivity=tail_table(ref,full,shape,risk))
    return phase,outputs,decision(intervals,point,sorted(set(p)),cfg)

def write_market_stop(run_dir,output,exception,phase):
    """No optimizer claim: preserve the first canonical scientific failure."""
    stop=ROOT/"outputs"/"MARKET_STOP.json"
    stop.parent.mkdir(parents=True,exist_ok=True)
    record=dict(stage="analysis",phase=phase,run=str(run_dir),output=str(output),
        error_type=type(exception).__name__,error=str(exception),
        traceback=traceback.format_exc(),real_fits_in_analysis=0,
        partial_outputs_preserved=True)
    try:
        with stop.open("x",encoding="utf-8") as stream:
            stream.write(json.dumps(clean_json(record),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
            stream.flush();os.fsync(stream.fileno())
    except FileExistsError:
        pass  # An earlier failure remains authoritative; never overwrite it.

def run(run_dir,output):
    # CLI/metadata preflight is outside scientific computation and cannot lock
    # the market stage merely because the requested command was invalid.
    run_dir=Path(run_dir).resolve();output=Path(output).resolve()
    canonical=(ROOT/"outputs").resolve()
    if not run_dir.is_relative_to(canonical) or not output.is_relative_to(canonical):
        raise ValueError("Input run and output must be children of canonical V4 outputs")
    if run_dir==canonical or output==canonical or output==run_dir:
        raise ValueError("Distinct child run and analysis directories required")
    if output.exists():raise FileExistsError("Analysis output must be new")
    protocol=ROOT/"FROZEN_PROTOCOL.md";config=ROOT/"frozen_config.json"
    if not protocol.exists() or not config.exists():raise ValueError("Final frozen protocol/config required for market analysis")
    status=json.loads((run_dir/"status.json").read_text(encoding="utf-8"))
    if status.get("status")!="COMPLETE" or (run_dir/"FAILED.json").exists():raise ValueError("Only COMPLETE runs")
    cfg=validate_config(json.loads(config.read_text(encoding="utf-8")))
    training_manifest_path=run_dir/"manifest.json"
    training_manifest=json.loads(training_manifest_path.read_text(encoding="utf-8"))
    if training_manifest.get("config_sha256")!=sha(config):raise ValueError("Training config identity differs")
    input_path=run_dir/"predictions_seeds.csv"
    started=False
    try:
        with ScientificClock("analysis:"+run_dir.name) as clock:
            started=True
            frame=pd.read_csv(input_path,dtype={"fold":str,"period":str})
            phase,outputs,dec=analyze_frames(frame,cfg,clock.check)
            clock.check();output.mkdir(parents=True,exist_ok=False)
            for name,table in outputs.items():table.to_csv(output/(name+".csv"),index=False)
            manifest=dict(phase=phase,input_sha256=sha(input_path),analysis_sha256=sha(__file__),
                training_manifest_sha256=sha(training_manifest_path),
                protocol_sha256=sha(protocol),config_sha256=sha(config),real_fits=0,
                inference=cfg["inference"],projections={k:v.tolist() for k,v in PROJECTIONS.items()},
                risk_ratio="mean of period ratios, never ratio of macro means",
                undefined_bootstrap="missing interval if any requested replicate undefined; no refilling",
                quadratic_roundoff="Only PSD-tolerance tiny negative aSa set to zero and counted",
                versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
                evidence=cfg["evidence_level_china" if phase=="exploration" else "evidence_level_external"])
            for name,value in [("continuation_decision",dec),("analysis_manifest",manifest)]:
                (output/(name+".json")).write_text(json.dumps(clean_json(value),indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
            clock.check()
    except BaseException as exception:
        if started:
            try:write_market_stop(run_dir,output,exception,training_manifest.get("phase"))
            except BaseException as stop_exception:
                # Keep the original scientific exception; make a failed fence
                # write explicit rather than reporting that a stop was saved.
                print("Could not persist canonical analysis MARKET_STOP: "+repr(stop_exception),file=sys.stderr)
        raise
    return dec

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=Path);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/"outputs"):raise ValueError("Output outside V4")
    print(json.dumps(clean_json(run(args.run,args.output)),indent=2,ensure_ascii=False))
