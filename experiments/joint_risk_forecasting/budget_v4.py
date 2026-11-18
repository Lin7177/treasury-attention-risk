"""Append-only V4 attempt journal; no old budget is touched."""
from pathlib import Path
import hashlib,json,os,time,math
LIMITS={"exploration":{"garch":9,"dcc":3,"har":9,"neural":27},
        "confirmation":{"garch":3,"dcc":1,"har":3,"neural":9}}
ROOT=Path(__file__).resolve().parent
JOURNAL=ROOT/"outputs/fit_budget.jsonl"
MAX_SECONDS=14400

def canonical(row):return json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def load(path=JOURNAL):
    path=Path(path)
    if not path.exists():return []
    rows=[json.loads(v) for v in path.read_text(encoding="utf-8").splitlines() if v]
    previous="0"*64
    for i,row in enumerate(rows,1):
        current=dict(row);digest=current.pop("event_sha256")
        if row["sequence"]!=i or row["previous_sha256"]!=previous or hashlib.sha256(canonical(current).encode()).hexdigest()!=digest:
            raise RuntimeError("Budget journal identity/sequence failure")
        previous=digest
    return rows
def append(row,path=JOURNAL):
    path=Path(path);rows=load(path)
    event=dict(row,sequence=len(rows)+1,previous_sha256=rows[-1]["event_sha256"] if rows else "0"*64)
    event["event_sha256"]=hashlib.sha256(canonical(event).encode()).hexdigest()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8",newline="\n") as f:
        f.write(canonical(event)+"\n");f.flush();os.fsync(f.fileno())
    return event
def state(path=JOURNAL):
    rows=load(path);jobs={};elapsed=0
    for row in rows:
        if row["event"]=="claim":
            if row["job"] in jobs:raise RuntimeError("Duplicate market attempt")
            jobs[row["job"]]=dict(row,status="pending")
        elif row["event"]=="finish":
            if row["job"] not in jobs or jobs[row["job"]]["status"]!="pending":raise RuntimeError("Invalid market completion")
            duration=row["seconds"]
            if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<0:
                raise RuntimeError("Nonfinite/invalid journal duration")
            jobs[row["job"]]["status"]=row["status"];elapsed+=duration
        else:raise RuntimeError("Unknown budget event")
    return jobs,elapsed
def claim(job,phase,kind,version,path=JOURNAL):
    if phase not in LIMITS or kind not in LIMITS[phase]:raise ValueError("Unregistered fit category")
    if Path(path).resolve()==JOURNAL.resolve() and phase=="confirmation":
        auth=ROOT/"DATA_AUTHORIZATION.json"
        if not auth.exists() or not json.loads(auth.read_text(encoding="utf-8-sig")).get("user_approved"):
            raise PermissionError("Public source/confirmation authorization not recorded")
    jobs,elapsed=state(path)
    if job in jobs:raise RuntimeError("Spent attempt cannot be retried")
    if any(v["status"]!="complete" for v in jobs.values()):raise RuntimeError("Pending/failed attempt blocks later market fits")
    used=sum(v["phase"]==phase and v["kind"]==kind for v in jobs.values())
    if used>=LIMITS[phase][kind] or len(jobs)>=64:raise RuntimeError("V4 finite attempt budget exhausted")
    if elapsed>=MAX_SECONDS:raise RuntimeError("Four-hour market optimization clock exhausted")
    return append(dict(event="claim",job=job,phase=phase,kind=kind,version=version),path)
def _safe_failure(value,location="",nonfinite=None):
    if nonfinite is None:nonfinite=[]
    if isinstance(value,float) and not math.isfinite(value):
        nonfinite.append(location or "<root>");return None
    if isinstance(value,dict):return {str(k):_safe_failure(v,location+"."+str(k),nonfinite) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_safe_failure(v,location+"[]",nonfinite) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if hasattr(value,"item"):return _safe_failure(value.item(),location,nonfinite)
    return str(value)
def finish(job,status,seconds,details=None,path=JOURNAL):
    if status not in ["complete","failed"]:raise ValueError("Invalid attempt status")
    jobs,_=state(path)
    if job not in jobs or jobs[job]["status"]!="pending":raise RuntimeError("Only own pending attempt may finish")
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0:raise ValueError("Invalid duration")
    payload=details or {}
    if status=="failed":
        nonfinite=[];payload=_safe_failure(payload,nonfinite=nonfinite)
        if nonfinite:payload={"original_diagnostics":payload,"nonfinite_diagnostic_paths":nonfinite,
                              "note":"Nonfinite failure diagnostics stored as null; failure is not discarded."}
    return append(dict(event="finish",job=job,status=status,seconds=float(seconds),details=payload),path)
def summary(path=JOURNAL):
    jobs,elapsed=state(path)
    return dict(claims=len(jobs),complete=sum(v["status"]=="complete" for v in jobs.values()),
        failed=sum(v["status"]=="failed" for v in jobs.values()),pending=sum(v["status"]=="pending" for v in jobs.values()),
        seconds=elapsed,maximum=64,phase_limits=LIMITS)
