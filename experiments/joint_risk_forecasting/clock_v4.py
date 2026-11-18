"""Serial, append-only real-data scientific wall clock. Synthetic tests use temporary ledgers."""
from pathlib import Path
from contextlib import contextmanager
import json,math,time,os,uuid
ROOT=Path(__file__).resolve().parent
PATH=ROOT/"outputs/compute_clock.jsonl"
LIMIT=14400.
def _rows(path):
    path=Path(path)
    return [json.loads(v) for v in path.read_text(encoding="utf-8").splitlines() if v] if path.exists() else []
def _append(row,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
def spent(path=PATH):
    starts={};seconds=0.
    for row in _rows(path):
        if row["event"]=="start":
            if row["token"] in starts:raise RuntimeError("Duplicate clock start")
            starts[row["token"]]=row
        elif row["event"]=="finish":
            if row["token"] not in starts:raise RuntimeError("Clock completion without start")
            starts.pop(row["token"]);value=row["seconds"]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise RuntimeError("Invalid scientific clock duration")
            seconds+=value
        else:raise RuntimeError("Unknown scientific clock event")
    return seconds,starts
class ScientificClock:
    def __init__(self,task,path=PATH,limit=LIMIT):
        self.path=Path(path);self.task=task;self.limit=limit
    def __enter__(self):
        used,pending=spent(self.path)
        if pending:raise RuntimeError("Unclosed scientific task blocks new real-data computation")
        if used>=self.limit:raise RuntimeError("Scientific compute envelope exhausted")
        self.token=uuid.uuid4().hex;self.start=time.perf_counter();self.deadline=self.start+self.limit-used
        self.lock=self.path.with_suffix(".active")
        self.lock.parent.mkdir(parents=True,exist_ok=True)
        with self.lock.open("x",encoding="utf-8") as f:f.write(self.token)
        try:_append(dict(event="start",task=self.task,token=self.token,seconds_used_before=used),self.path)
        except BaseException:
            if self.lock.read_text()==self.token:self.lock.unlink()
            raise
        return self
    def check(self):
        if time.perf_counter()>self.deadline:raise TimeoutError("Four-hour real-data scientific wall clock exceeded")
    def __exit__(self,kind,value,traceback):
        elapsed=time.perf_counter()-self.start
        _append(dict(event="finish",task=self.task,token=self.token,seconds=elapsed,status="complete" if kind is None else "failed"),self.path)
        if self.lock.exists() and self.lock.read_text()==self.token:self.lock.unlink()
        return False
