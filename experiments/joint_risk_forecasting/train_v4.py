"""Bounded V4 execution. Import is inert; CLI requires frozen gate and authorization."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
for _thread_variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable]="1"
from pathlib import Path
import platform
import re
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

import budget_v4 as budget
import clock_v4
import data_v4 as data
import financial_v4 as financial
from models_v4 import build_model, fixed_matrix, kernel_bandwidth_statistics

ROOT = Path(__file__).resolve().parent
US_COLUMNS = ["Date","1 Yr","3 Yr","5 Yr","7 Yr","10 Yr"]
UPPER = [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
PRED_FIELDS = ["fold","phase","role","model","seed","index","date","information_end","label_end",
               "y00","y01","y02","y11","y12","y22","f00","f01","f02","f11","f12","f22",
               "common_d0","common_d1","common_d2","period"]
FIT_FIELDS = ["job","phase","fold","kind","model","seed","parameters","selected_epoch","iterations",
              "seconds","optimizer_seconds","state_path","initialization_sha256","status"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_safe(value, path="$", issues=None):
    """Replace nonfinite diagnostics with null and retain explicit reasons."""
    if issues is None:
        issues = []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            issues.append({"path": path, "reason": f"nonfinite {value}"})
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [json_safe(v, f"{path}[{i}]", issues) for i,v in enumerate(value)]
    if isinstance(value, dict):
        return {str(k): json_safe(v, f"{path}.{k}", issues) for k,v in value.items()}
    issues.append({"path": path, "reason": f"unsupported diagnostic type {type(value).__name__}"})
    return str(value)


def save_new(path, value):
    issues = []
    clean = json_safe(value, issues=issues)
    if issues:
        clean = dict(clean, diagnostic_sanitization=issues) if isinstance(clean, dict) else {
            "value": clean, "diagnostic_sanitization": issues}
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(clean, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def append_rows(path, rows, columns=None):
    frame = pd.DataFrame(rows, columns=columns)
    if not len(frame):
        return
    path = Path(path)
    first = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, header=first, float_format="%.17g")
        stream.flush()
        os.fsync(stream.fileno())


def git_version(root=ROOT):
    result = subprocess.run(["git","rev-parse","HEAD"], cwd=root, capture_output=True, text=True)
    version = result.stdout.strip()
    if result.returncode or not re.fullmatch("[0-9a-fA-F]{40}", version):
        raise RuntimeError("A real Git HEAD is required before market execution")
    return version.lower()


def validate_launch(root=ROOT):
    root = Path(root)
    if (root/"outputs/MARKET_STOP.json").exists():
        raise RuntimeError("Prior scientific failure permanently blocks this frozen market stage")
    freeze = read_json(root/"FREEZE_MANIFEST.json")
    gate = read_json(root/"outputs/technical_gate.json")
    if not gate.get("passed") or gate.get("freeze_manifest_sha256") != sha(root/"FREEZE_MANIFEST.json"):
        raise RuntimeError("Technical gate is absent, failed, or bound to another freeze")
    reviews = freeze.get("independent_final_reviews", [])
    if len(reviews) < 2 or len({r.get("role") for r in reviews}) < 2:
        raise RuntimeError("Two independent final reviews are required")
    for review in reviews:
        if review.get("status") != "PASS" or sha(root/review["path"]) != review["sha256"]:
            raise RuntimeError("Final review identity/status mismatch")
    for name, expected in freeze["sha256"].items():
        if sha(root/name) != expected:
            raise RuntimeError("Frozen material changed: " + name)
    runtime = gate.get("implementation_sha256", {})
    required = {"train_v4.py","models_v4.py","data_v4.py","financial_v4.py","budget_v4.py","clock_v4.py"}
    if not required <= set(runtime):
        raise RuntimeError("Incomplete runtime gate")
    for name, expected in runtime.items():
        if sha(root/name) != expected:
            raise RuntimeError("Runtime changed after technical gate: " + name)
    cfg = read_json(root/"frozen_config.json")
    if not isinstance(cfg.get("epochs"), int) or cfg["epochs"] != 40:
        raise RuntimeError("Unexpected frozen training schedule")
    return cfg, freeze, gate, git_version(root)


def check_us_authorization(cfg, auth_path):
    auth = read_json(auth_path)
    ext = cfg["external"]
    if auth.get("user_approved") is not True:
        raise PermissionError("Explicit public-source numeric authorization is absent")
    for key, expected in (("source",ext["source"]),("date_min",ext["date_min"]),
                          ("date_max",ext["date_max"]),("columns",US_COLUMNS)):
        if auth.get(key) != expected:
            raise PermissionError("External source authorization scope mismatch: " + key)
    return auth


def load_us(source, cfg, auth_path):
    auth = check_us_authorization(cfg, auth_path)  # Before source hash or numeric read.
    digest = data.file_sha256(source)
    text = pd.read_csv(source, usecols=US_COLUMNS, dtype=str, keep_default_na=False)
    dates = pd.to_datetime(text["Date"].str.strip().str.slice(0,10), format="%m/%d/%Y", errors="raise")
    ext = cfg["external"]
    keep = (dates >= pd.Timestamp(ext["date_min"])) & (dates <= pd.Timestamp(ext["date_max"]))
    prefix = text.loc[keep].copy()
    axis, levels, metadata = data.parse_us_text_frame(prefix, ext["date_max"])
    result = data.build_dataset(axis, levels, cutoff=ext["date_max"])
    result[0].attrs.update(source_sha256=digest, parse_metadata=metadata,
                           authorization_sha256=sha(auth_path), authorization_scope=auth)
    return result


def split_us(d, cfg):
    ext = cfg["external"]
    ev = np.flatnonzero((d.date.dt.year >= ext["evaluation_first_year"]) &
                        (d.date.dt.year <= ext["evaluation_last_year"]) &
                        (d.label_end <= pd.Timestamp(ext["date_max"])))
    if not len(ev):
        raise ValueError("No external confirmation requests")
    va = np.flatnonzero((d.date.dt.year >= ext["validation_first_year"]) &
                        (d.date.dt.year <= ext["validation_last_year"]) &
                        (d.label_end < d.iloc[ev[0]].date))
    if not len(va):
        raise ValueError("No external validation requests")
    tr = np.flatnonzero((d.date.dt.year <= ext["train_last_request_year"]) &
                        (d.label_end < d.iloc[va[0]].date))
    counts = tuple(map(len,(tr,va,ev)))
    if any(n < ext[k] for n,k in zip(counts,("min_train","min_validation","min_evaluation"))):
        raise ValueError("External coverage below frozen minimum")
    if set(tr)&set(va) or set(tr)&set(ev) or set(va)&set(ev):
        raise ValueError("External roles overlap")
    assert d.iloc[tr].label_end.max() < d.iloc[va[0]].date
    assert d.iloc[va].label_end.max() < d.iloc[ev[0]].date
    return tr,va,ev


def period_for(date, phase, cfg):
    year = pd.Timestamp(date).year
    if phase == "exploration":
        return str(year)
    for lo,hi in cfg["external"]["subperiods"]:
        if lo <= year <= hi:
            return f"{lo}-{str(hi)[-2:]}"
    return "US_preconfirmation"


def tensor_qlike(target, forecast):
    chol = torch.linalg.cholesky(forecast)
    determinant = 2*torch.log(torch.diagonal(chol,dim1=-2,dim2=-1)).sum(-1)
    trace = torch.diagonal(torch.cholesky_solve(target,chol),dim1=-2,dim2=-1).sum(-1)
    values = (determinant + trace)/3
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite neural matrix QLIKE")
    return values.mean()


class AuditRunner:
    """Internal executor; public run() alone establishes market launch/clock guard."""
    def __init__(self,out,cfg,phase,version,clock,journal=budget.JOURNAL,fin=financial):
        self.out,self.cfg,self.phase,self.version = Path(out),cfg,phase,version
        self.clock,self.journal,self.fin = clock,Path(journal),fin
        self.active = None
        self.active_start = None
        self.claimed = 0
        self.optimizer_seconds = 0.0
        self.active_optimizer_start = None
        self.last_optimizer_seconds = 0.0
        self.active_state = None
        self.active_model = None
        self.active_fold = None
        for folder in ("checkpoints","predictions"):
            (self.out/folder).mkdir()
    def claim(self,job,kind):
        self.clock.check()
        budget.claim(job,self.phase,kind,self.version,path=self.journal)
        self.active,self.active_start = job,time.perf_counter()
        self.claimed += 1
        self.last_optimizer_seconds=0.0
    def stop_optimizer_clock(self):
        if self.active_optimizer_start is not None:
            elapsed=time.perf_counter()-self.active_optimizer_start
            self.last_optimizer_seconds+=elapsed
            self.optimizer_seconds+=elapsed
            self.active_optimizer_start=None
    def finish(self,details):
        details=dict(details,optimizer_seconds=self.last_optimizer_seconds)
        self.clock.check()
        budget.finish(self.active,"complete",time.perf_counter()-self.active_start,
                      details=json_safe(details),path=self.journal)
        self.active,self.active_start,self.active_model,self.active_state = None,None,None,None
    def component(self,fold,kind,coordinate,call,parameters):
        job=f"{self.phase}_{fold}_{kind}_{coordinate}"
        self.claim(job,kind)
        self.active_optimizer_start=time.perf_counter()
        state=call()
        self.stop_optimizer_clock()
        self.active_state=state
        self.clock.check()
        file=self.out/"checkpoints"/f"{job}.json"
        save_new(file,state)
        rows=[]
        for row in state.get("trace",[]):
            rows.append(dict(phase=self.phase,fold=str(fold),job=job,kind=kind,coordinate=coordinate,**row))
        append_rows(self.out/"financial_trace.csv",rows,
                    ["phase","fold","job","kind","coordinate","iteration","loss","projected_gradient","gradient_inf"])
        seconds=time.perf_counter()-self.active_start
        append_rows(self.out/"fits.csv",[dict(job=job,phase=self.phase,fold=str(fold),kind=kind,
            model=f"{kind.upper()}_{coordinate}",seed=0,parameters=parameters,selected_epoch=None,
            iterations=state.get("iterations"),seconds=seconds,optimizer_seconds=self.last_optimizer_seconds,state_path=str(file.relative_to(self.out)),
            initialization_sha256=None,status="complete")],FIT_FIELDS)
        self.finish({"iterations":state.get("iterations"),"state_sha256":sha(file)})
        return state
    def record(self,fold,model,seed,role,ix,d,targets,forecast,common):
        self.clock.check()
        forecast=np.asarray(forecast,dtype=np.float64)
        y=np.asarray(targets[ix],dtype=np.float64)
        if forecast.shape != y.shape:
            raise ValueError("Prediction shape or common-role denominator mismatch")
        scores=self.fin.matrix_qlike(y,forecast)
        frame=d.iloc[ix][["index","date","information_end","label_end"]].copy()
        frame["fold"],frame["phase"],frame["role"] = str(fold),self.phase,role
        frame["model"],frame["seed"] = model,seed
        for i,j in UPPER:
            frame[f"y{i}{j}"]=y[:,i,j]
            frame[f"f{i}{j}"]=forecast[:,i,j]
        for i in range(3):
            frame[f"common_d{i}"]=common[ix,i]
        frame["period"]=[period_for(v,self.phase,self.cfg) for v in frame.date]
        file=self.out/"predictions"/f"{fold}_{model}_{seed}_{role}.csv"
        if file.exists():
            raise FileExistsError(file)
        with file.open("x",encoding="utf-8",newline="") as stream:
            frame[PRED_FIELDS].to_csv(stream,index=False,float_format="%.17g")
            stream.flush();os.fsync(stream.fileno())
        append_rows(self.out/"predictions_seeds.csv",frame[PRED_FIELDS],PRED_FIELDS)
        append_rows(self.out/"role_metrics.csv",[dict(fold=str(fold),phase=self.phase,role=role,
            model=model,seed=seed,n=len(ix),physical_matrix_qlike=float(scores.mean()),
            minimum_cholesky_diagonal=float(np.diagonal(np.linalg.cholesky(forecast),axis1=-2,axis2=-1).min()))])
        self.clock.check()
    def run_fold(self,fold,roles,d,rawx,targets,axis_dates,raw_delta):
        self.clock.check()
        self.active_fold=str(fold)
        tr,va,ev=roles["train"],roles["validation"],roles["evaluation"]
        statistics=data.fit_training_statistics(raw_delta,d,tr)
        rms=np.asarray(statistics["rms5"],dtype=float)[2:5]
        c0=np.asarray(statistics["C0"],dtype=float)
        x=torch.tensor(data.normalize_inputs(rawx,statistics),dtype=torch.float64)
        y=torch.tensor(data.normalize_targets(targets,statistics),dtype=torch.float64)
        scale_outer=rms[:,None]*rms[None,:]
        bandwidth=kernel_bandwidth_statistics(x[tr])
        save_new(self.out/f"statistics_{fold}.json",dict(**statistics,
            train_positions=tr.tolist(),validation_positions=va.tolist(),evaluation_positions=ev.tolist(),
            kernel=bandwidth,training_statistics_not_optimizers=True))
        append_rows(self.out/"statistics_estimates.csv",[dict(fold=str(fold),kind="RMS5_Ctrain_C0",
            training_requests=len(tr),unique_days=statistics["unique_day_count"],optimizer_fits=0),
            dict(fold=str(fold),kind="kernel_geometry",training_requests=len(tr),
                 unique_days=None,optimizer_fits=0)])
        w=np.exp(np.asarray(self._time_log_prior()))
        common=5*(rawx[:,:,2:5].square() if hasattr(rawx,"square") else np.square(rawx[:,:,2:5]))*w[None,:,None]
        common=common.sum(axis=1)
        floor=1e-8*rms**2
        counts={r:np.sum(common[ix]<=floor,axis=0).tolist() for r,ix in roles.items()}
        common=np.maximum(common,floor)
        if not np.isfinite(common).all():
            raise FloatingPointError("Nonfinite common diagonal")
        save_new(self.out/f"common_diagonal_{fold}.json",dict(floor=floor.tolist(),trigger_counts=counts,
                  definition="5*finite20_EWMA(raw target delta squared), floor=1e-8*rms_target_squared"))
        append_rows(self.out/"sample_flow.csv",[dict(fold=str(fold),phase=self.phase,train=len(tr),
            validation=len(va),evaluation=len(ev),train_label_end=str(d.iloc[tr].label_end.max()),
            validation_start=str(d.iloc[va].date.min()),validation_label_end=str(d.iloc[va].label_end.max()),
            evaluation_start=str(d.iloc[ev].date.min()),unique_input_days=statistics["unique_day_count"])])
        for name in ("Prior","Uniform","EWMA","Kernel"):
            for role,ix in roles.items():
                self.clock.check()
                with torch.no_grad():
                    h=(5*torch.tensor(c0,dtype=torch.float64).expand(len(ix),-1,-1) if name=="Prior" else
                       fixed_matrix(x[ix],c0,name,bandwidth_squared=bandwidth["bandwidth_squared"]))
                self.record(fold,name,0,role,ix,d,targets,h.numpy()*scale_outer,common)
        # Traditional filters use the continuous original axis, preserving missing dates.
        train_axis=data.training_axis_indices(d,tr,len(axis_dates))
        start=int(train_axis[0])
        last=max(int(d.iloc[ix]["index"].max()) for ix in roles.values())
        raw_r=raw_delta[start:last+1,2:5]/rms
        training_r=raw_delta[train_axis,2:5]/rms
        gp=[]
        for coordinate,label in enumerate(("5Y","7Y","10Y")):
            state=self.component(fold,"garch",label,
                lambda j=coordinate:self.fin.fit_garch(training_r[:,j],deadline=self.clock.deadline),3)
            gp.append(state["parameters"])
        self.clock.check()
        haxis=np.column_stack([self.fin.garch_filter(raw_r[:,j],**p) for j,p in enumerate(gp)])
        if not np.isfinite(haxis).all() or (haxis<=0).any():
            raise FloatingPointError("Nonfinite filtered GARCH variance")
        innovations=raw_r/np.sqrt(haxis)
        dcc_state=self.component(fold,"dcc","joint",
            lambda:self.fin.fit_dcc(innovations[:len(train_axis)],deadline=self.clock.deadline),2)
        self.clock.check()
        filtered=self.fin.dcc_filter(innovations,np.asarray(dcc_state["qbar"]),
                                    dcc_state["parameters"]["alpha"],dcc_state["parameters"]["beta"])
        qaxis=filtered["q"]
        rccc=self.fin.correlation(np.asarray(dcc_state["qbar"]))
        har=[]
        for coordinate,label in enumerate(("5Y","7Y","10Y")):
            state=self.component(fold,"har",label,
                lambda j=coordinate:self.fin.fit_har(rawx[tr,:,j+2],targets[tr,j,j],
                                                    deadline=self.clock.deadline),4)
            har.append(state)
        for role,ix in roles.items():
            self.clock.check()
            positions=d.iloc[ix]["index"].to_numpy(dtype=int)-start
            current=haxis[positions].copy()
            aggregate=np.zeros_like(current)
            omega=np.array([p["omega"] for p in gp])
            a=np.array([p["alpha"] for p in gp])
            b=np.array([p["beta"] for p in gp])
            for _ in range(5):
                aggregate+=current
                current=omega+(a+b)*current
            roots=np.sqrt(aggregate)
            gccc=rccc[None,:,:]*roots[:,:,None]*roots[:,None,:]
            self.record(fold,"GARCH_CCC",0,role,ix,d,targets,gccc*scale_outer,common)
            fhar=np.column_stack([self.fin.predict_har(rawx[ix,:,j+2],state) for j,state in enumerate(har)])
            roots=np.sqrt(fhar)
            hhar=c0[None,:,:]*roots[:,:,None]*roots[:,None,:]
            self.record(fold,"HAR_CCC",0,role,ix,d,targets,hhar,common)
            parts=[]
            for offset in range(0,len(ix),8):
                self.clock.check()
                batch=positions[offset:offset+8]
                result,diag=self.fin.controlled_dcc_five(haxis[batch],qaxis[batch],gp,dcc_state,
                                                       seed=self.cfg["qmc"]["seed"])
                self.clock.check()
                parts.append(result)
                diagnostic=[]
                for j,row_position in enumerate(ix[offset:offset+8]):
                    diagnostic.append(dict(fold=str(fold),phase=self.phase,role=role,
                        index=int(d.iloc[row_position]["index"]),date=str(d.iloc[row_position].date),
                        paths=int(diag["paths"][j]),stein=float(diag["stein"][j]),
                        relative_frobenius=float(diag["relative_frobenius"][j]),
                        upgraded=bool(diag["paths"][j]>8192)))
                append_rows(self.out/"qmc_precision.csv",diagnostic)
            self.record(fold,"DCC",0,role,ix,d,targets,np.concatenate(parts)*scale_outer,common)
        for name in self.cfg["learned"]:
            for seed in self.cfg["seeds"]:
                self.neural(fold,name,seed,roles,d,targets,x,y,c0,scale_outer,common)
        self.active_fold=None
        self.clock.check()

    @staticmethod
    def _time_log_prior():
        from models_v4 import time_log_prior
        return time_log_prior().numpy()

    def neural(self,fold,name,seed,roles,d,targets,x,y,c0,scale_outer,common):
        tr,va=roles["train"],roles["validation"]
        job=f"{self.phase}_{fold}_{name}_{seed}"
        self.claim(job,"neural")
        model=build_model(name,seed)
        self.active_model=model
        init=model.initialization_sha256()
        save_new(self.out/"checkpoints"/f"{job}_initialization.json",dict(
            name=name,seed=seed,fold=str(fold),sha256=init,parameters=model.parameter_summary()))
        opt=torch.optim.AdamW(model.parameters(),lr=self.cfg["lr"],weight_decay=self.cfg["weight_decay"])
        rng=np.random.default_rng(seed)
        best=float("inf")
        best_state=None
        best_epoch=None
        c0tensor=torch.tensor(c0,dtype=torch.float64)
        self.active_optimizer_start=time.perf_counter()
        for epoch in range(1,self.cfg["epochs"]+1):
            model.train()
            order=rng.permutation(tr)
            online=0.
            maxgrad=0.
            for offset in range(0,len(order),self.cfg["batch"]):
                self.clock.check()
                ids=order[offset:offset+self.cfg["batch"]]
                opt.zero_grad()
                value=tensor_qlike(y[ids],model(x[ids],c0tensor))
                value.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),self.cfg["gradient_clip"])
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite neural task gradient")
                opt.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite neural parameter after optimizer step")
                online+=float(value.detach())*len(ids)
                maxgrad=max(maxgrad,float(norm.detach()))
                self.clock.check()
            model.eval()
            with torch.no_grad():
                fixed_train=float(tensor_qlike(y[tr],model(x[tr],c0tensor)))
                valid=float(tensor_qlike(y[va],model(x[va],c0tensor)))
            self.clock.check()
            append_rows(self.out/"training_trace.csv",[dict(fold=str(fold),phase=self.phase,model=name,seed=seed,
                epoch=epoch,online_normalized_matrix_qlike=online/len(tr),
                fixed_train_normalized_matrix_qlike=fixed_train,
                validation_normalized_matrix_qlike=valid,max_unclipped_grad=maxgrad)])
            if valid<best:
                best,best_epoch,best_state=valid,epoch,copy.deepcopy(model.state_dict())
        self.stop_optimizer_clock()
        if best_state is None:
            raise RuntimeError("No finite validation checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        checkpoint=self.out/"checkpoints"/f"{job}.pt"
        with checkpoint.open("xb") as stream:
            torch.save(best_state,stream)
            stream.flush();os.fsync(stream.fileno())
        for role,ix in roles.items():
            self.clock.check()
            with torch.no_grad():
                h=model(x[ix],c0tensor).numpy()*scale_outer
            self.record(fold,name,seed,role,ix,d,targets,h,common)
        seconds=time.perf_counter()-self.active_start
        append_rows(self.out/"fits.csv",[dict(job=job,phase=self.phase,fold=str(fold),kind="neural",
            model=name,seed=seed,parameters=model.parameter_summary()["total"],
            selected_epoch=best_epoch,iterations=self.cfg["epochs"],seconds=seconds,optimizer_seconds=self.last_optimizer_seconds,
            state_path=str(checkpoint.relative_to(self.out)),initialization_sha256=init,status="complete")],FIT_FIELDS)
        self.finish(dict(selected_epoch=best_epoch,validation_normalized_matrix_qlike=best,
                         checkpoint_sha256=sha(checkpoint)))

    def fail(self,exc):
        self.stop_optimizer_clock()
        details=dict(error=str(exc),error_type=type(exc).__name__,active_job=self.active,
                     active_fold=self.active_fold,returned_component_state=self.active_state,fit_trace=getattr(exc,"trace",[]),
                     diagnostics=getattr(exc,"details",getattr(exc,"diagnostics",{})))
        errors=[]
        if self.active_model is not None:
            try:
                path=self.out/"FAILED_CURRENT_STATE.pt"
                with path.open("xb") as stream:
                    torch.save(self.active_model.state_dict(),stream)
                    stream.flush();os.fsync(stream.fileno())
                details["failed_current_state"]=str(path.relative_to(self.out))
            except BaseException as write_error:
                errors.append("Cannot persist failed neural state: "+str(write_error))
        if self.active is not None:
            try:
                budget.finish(self.active,"failed",time.perf_counter()-self.active_start,
                              details=json_safe({"error":str(exc),"optimizer_seconds":self.last_optimizer_seconds}),path=self.journal)
            except BaseException as finish_error:
                errors.append("Cannot close pending consumed attempt: "+str(finish_error))
        details["failure_persistence_errors"]=errors
        try:
            details["budget"]=budget.summary(self.journal)
        except BaseException as summary_error:
            details["budget"]=None
            errors.append("Cannot read consumed attempt journal: "+str(summary_error))
        try:
            save_new(self.out/"FAILED.json",details)
        finally:
            if self.journal.resolve()==budget.JOURNAL.resolve():
                stop=ROOT/"outputs/MARKET_STOP.json"
                if not stop.exists():
                    save_new(stop,dict(run=str(self.out),phase=self.phase,error=str(exc),
                                       consumed_attempts_remain_spent=True))

def run(phase,source,output):
    if phase not in ("exploration","confirmation"):
        raise ValueError("Unknown phase")
    cfg,freeze,gate,version=validate_launch()
    output=Path(output).resolve()
    if not output.is_relative_to((ROOT/"outputs").resolve()) or output==(ROOT/"outputs").resolve():
        raise ValueError("Run output must be a new child of this V4 outputs directory")
    if output.exists():
        raise FileExistsError("Output directories cannot be reused")
    if phase=="confirmation":
        check_us_authorization(cfg,ROOT/"DATA_AUTHORIZATION.json")
    existing,_=budget.state()
    if any(v["status"]!="complete" for v in existing.values()):
        raise RuntimeError("Failed or pending market attempt blocks this stage")
    if any(v["phase"]==phase for v in existing.values()):
        raise RuntimeError("This phase already consumed attempts and cannot be repeated")
    before_clock,_=clock_v4.spent()
    runner=None
    science_started=False
    output.mkdir(parents=True,exist_ok=False)
    save_new(output/"launch.json",dict(phase=phase,git_commit=version,
             started_utc=datetime.now(timezone.utc).isoformat(),output=str(output)))
    try:
        with clock_v4.ScientificClock(f"V4_{phase}") as clock:
            science_started=True
            clock.check()
            if phase=="exploration":
                if data.file_sha256(source)!=cfg["source_china_sha256"]:
                    raise ValueError("Chinese source identity mismatch")
                dataset=data.load_data(source)
                data.verify_local_counts(dataset[0])
                plans=[(str(year),data.split_data(dataset[0],year)) for year in cfg["china_years"]]
            else:
                dataset=load_us(source,cfg,ROOT/"DATA_AUTHORIZATION.json")
                plans=[("US",split_us(dataset[0],cfg))]
            clock.check()
            d,rawx,targets,axis_dates,raw_delta=dataset
            save_new(output/"manifest.json",dict(phase=phase,commit=version,
                source=str(Path(source).resolve()),source_sha256=data.file_sha256(source),
                authorization_sha256=sha(ROOT/"DATA_AUTHORIZATION.json") if phase=="confirmation" else None,
                freeze_manifest_sha256=sha(ROOT/"FREEZE_MANIFEST.json"),
                config_sha256=sha(ROOT/"frozen_config.json"),uv_lock_sha256=sha(ROOT/"uv.lock"),
                implementation_sha256=gate["implementation_sha256"],independent_final_reviews=freeze["independent_final_reviews"],
                python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
                device="cpu",dtype="float64",dataset_flow=d.attrs,
                evidence_level=cfg["evidence_level_china"] if phase=="exploration" else cfg["evidence_level_external"],
                model_names=cfg["models"],common_diagonal_units="5*raw target daily variance, physical bp squared",
                scientific_seconds_before=before_clock,
                garch_ccc_forecast="aggregate-scale constant-correlation plug-in, not exact multistep CCC expectation"))
            with (output/"sample_ledger.csv").open("x",encoding="utf-8",newline="") as stream:
                d.to_csv(stream,index=False);stream.flush();os.fsync(stream.fileno())
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            runner=AuditRunner(output,cfg,phase,version,clock)
            for fold,(tr,va,ev) in plans:
                roles={"train":tr,"validation":va,"evaluation":ev}
                runner.run_fold(fold,roles,d,rawx,targets,axis_dates,raw_delta)
            clock.check()
            expected=48 if phase=="exploration" else 16
            if runner.claimed!=expected:
                raise RuntimeError("Frozen unique optimizer plan is incomplete")
            jobs,_=budget.state()
            phase_jobs=[j for j in jobs.values() if j["phase"]==phase]
            if len(phase_jobs)!=expected or any(j["status"]!="complete" for j in phase_jobs):
                raise RuntimeError("Phase journal does not match complete frozen optimizer matrix")
            save_new(output/"budget_summary.json",budget.summary())
            with (output/"budget.jsonl").open("xb") as stream:
                stream.write(budget.JOURNAL.read_bytes());stream.flush();os.fsync(stream.fileno())
            clock.check()
        after_clock,pending=clock_v4.spent()
        if pending:
            raise RuntimeError("Scientific wall clock failed to close")
        summary=budget.summary()
        save_new(output/"status.json",dict(status="COMPLETE",phase=phase,
            phase_optimizer_attempts=48 if phase=="exploration" else 16,
            failures=0,budget=summary,
            optimizer_seconds_this_phase=runner.optimizer_seconds,
            optimizer_seconds_cumulative=sum(r.get("details",{}).get("optimizer_seconds",0.0)
                for r in budget.load() if r["event"]=="finish"),
            claimed_attempt_lifecycle_seconds_cumulative=summary["seconds"],
            optimizer_clock_definition="fit algorithm/checkpoint-selection loop wall time, includes epoch trace IO, excludes role prediction/statistics",
            scientific_seconds_this_phase=after_clock-before_clock,
            scientific_seconds_cumulative=after_clock,
            independent_observations_not_multiplied_by_models_or_seeds=True))
        return output
    except BaseException as exc:
        try:
            if runner is not None:
                runner.fail(exc)
            elif not (output/"FAILED.json").exists():
                save_new(output/"FAILED.json",dict(error=str(exc),error_type=type(exc).__name__,
                                                   market_optimizers_started_in_this_run=0))
        finally:
            # ETL/statistics/fixed-rule failures can occur before any optimizer.
            # They still terminate this registered scientific experiment.
            stop=ROOT/"outputs/MARKET_STOP.json"
            if science_started and not stop.exists():
                save_new(stop,dict(run=str(output),phase=phase,error=str(exc),
                    science_started=True,optimizer_attempts_started=runner.claimed if runner else 0,
                    consumed_attempts_remain_spent=True))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["exploration","confirmation"],required=True)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(args.phase,args.source,args.output)
