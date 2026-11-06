"""Frozen five-tenor, annual-forward mechanism study; source files are data only."""
import argparse,copy,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from models import MODEL_NAMES,build_model,parameter_counts
ROOT=Path(__file__).resolve().parent
COLS=['yield_1y','yield_3y','yield_5y','yield_7y','yield_10y']
SEEDS=[202611041,202611042,202611043]

def version_id():
    if (ROOT/'.git').exists() or (ROOT.parent/'.git').exists():
        try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError):pass
    path=ROOT/'SOURCE_VERSION.json'
    return json.loads(path.read_text(encoding='utf-8'))['commit'] if path.exists() else 'unversioned_archive'

def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sigmoid(z):return np.exp(-np.logaddexp(0.,-z))

def dataset(path):
    f=pd.read_csv(path,usecols=['date']+COLS,dtype=str)
    dates=pd.to_datetime(f.date,format='mixed',errors='raise').dt.strftime('%Y-%m-%d')
    f=f.loc[dates<='2024-06-30'].copy();dates=dates.loc[f.index].reset_index(drop=True)
    y=f[COLS].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    assert len(y)==3381 and dates.iloc[-1]=='2024-06-28'
    delta=np.vstack([np.full((1,5),np.nan),np.diff(y,axis=0)*100])
    rows=[];xs=[];ss=[]
    for t in range(21,len(y)-5):
        x=delta[t-20:t];level=y[t-1]
        if not np.isfinite(x).all() or not np.isfinite(level).all() or not np.isfinite(y[[t,t+5],3]).all():continue
        change=(y[t+5,3]-y[t,3])*100
        rows.append(dict(index=t,date=dates.iloc[t],label_end=dates.iloc[t+5],direction=int(np.sign(change)),change_bp=change))
        xs.append(x);ss.append([(level[4]-level[0])*100,(level[3]-.6*level[2]-.4*level[4])*100,x[:,3].std()])
    return pd.DataFrame(rows),np.asarray(xs),np.asarray(ss)

def split(d,year):
    ev=d.index[(d.date.str[:4]==str(year)) & (d.date<='2024-06-21') & (d.label_end<=f'{year}-12-31')].to_numpy()
    anchor=d.loc[ev,'date'].min();cal=d.index[d.label_end<anchor].to_numpy()[-126:]
    tr=d.index[d.label_end<d.loc[cal,'date'].min()].to_numpy()
    expected={2022:(237,8,1084,124),2023:(237,2,1322,120),2024:(112,2,1557,125)}[year]
    train=tr[d.loc[tr,'direction'].to_numpy()!=0];valid=cal[d.loc[cal,'direction'].to_numpy()!=0]
    assert (len(ev),int(d.loc[ev,'direction'].eq(0).sum()),len(train),len(valid))==expected
    assert d.loc[train,'label_end'].max()<d.loc[cal,'date'].min()
    return train,valid,ev,cal

def feature(x,s,own):
    a=x[:,:,3:4] if own else x
    v=np.concatenate([a[:,-1],a[:,-5:].sum(1),a.sum(1),a.std(1)],axis=1)
    return v if own else np.concatenate([v,s],axis=1)

def logistic(x,y):
    mu=x.mean(0);sd=x.std(0);sd[sd==0]=1;z=np.c_[np.ones(len(x)),(x-mu)/sd];w=np.zeros(z.shape[1])
    def obj(w):return np.mean(np.logaddexp(0,z@w)-y*(z@w))+.005*np.dot(w[1:],w[1:])
    for _ in range(100):
        p=sigmoid(z@w);g=z.T@(p-y)/len(y);g[1:]+=.01*w[1:]
        if np.abs(g).max()<1e-8:return mu,sd,w
        h=z.T@((p*(1-p))[:,None]*z)/len(y);h[1:,1:]+=.01*np.eye(len(w)-1)
        step=np.linalg.solve(h,g);rate=1.
        for j in range(30):
            if obj(w-rate*step)<=obj(w)-1e-4*rate*np.dot(g,step):w-=rate*step;break
            rate/=2
        else:raise RuntimeError('Logit line search failed')
    raise RuntimeError('Logit failed to converge')

def metrics(p,d):
    p=np.asarray(p);y=d.direction.to_numpy();pred=np.where(p>=.5,1,-1);nf=y!=0
    up=float((pred[y==1]==1).mean()) if (y==1).any() else None
    down=float((pred[y==-1]==-1).mean()) if (y==-1).any() else None
    pn=np.clip(p[nf],1e-12,1-1e-12);yn=(y[nf]==1).astype(float)
    return dict(n=len(y),nonflat_n=int(nf.sum()),accuracy=float((pred==y).mean()),balanced_accuracy=(up+down)/2 if up is not None and down is not None else None,up_recall=up,down_recall=down,predicted_up_share=float((pred==1).mean()),brier=float(np.mean((pn-yn)**2)) if len(pn) else None,log_loss=float(-np.mean(yn*np.log(pn)+(1-yn)*np.log1p(-pn))) if len(pn) else None)

def run(source,out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    d,x,s=dataset(source);d.to_csv(out/'sample_ledger.csv',index=False)
    save(out/'manifest.json',dict(source=str(source.resolve()),source_sha256=sha(source),code_sha256=sha(__file__),models_sha256=sha(ROOT/'models.py'),protocol_sha256=sha(ROOT/'PROTOCOL.md'),lock_sha256=sha(ROOT/'uv.lock'),torch=torch.__version__,numpy=np.__version__,seeds=SEEDS,commit=version_id(),device='cpu',training_epochs=20))
    predictions=[];traces=[];flows=[];params=[];completed=0
    try:
        for year in [2022,2023,2024]:
            tr,cal,ev,calall=split(d,year);labels=(d.loc[tr,'direction'].to_numpy()==1).astype(float)
            xm=x[tr].mean((0,1));xs=x[tr].std((0,1));xs[xs==0]=1
            sm=s[tr].mean(0);sd=s[tr].std(0);sd[sd==0]=1
            X=torch.tensor((x-xm)/xs,dtype=torch.float32);S=torch.tensor((s-sm)/sd,dtype=torch.float32)
            Y=torch.tensor((d.direction.to_numpy()==1).astype(float),dtype=torch.float32)
            volcut=float(np.median(s[tr,2]));flow=dict(year=year,train_n=len(tr),cal_nonflat_n=len(cal),cal_all_n=len(calall),eval_n=len(ev),train_label_end=d.loc[tr,'label_end'].max(),cal_start=d.loc[calall,'date'].min(),eval_start=d.loc[ev,'date'].min(),volcut=volcut)
            flows.append(flow);save(out/'sample_flow.json',flows)
            def record(name,seed,p):
                t=d.loc[ev].copy();t['model']=name;t['seed']=seed;t['probability_up']=p;t['vol_state']=np.where(s[ev,2]>=volcut,'high','low');predictions.append(t)
            record('DOWN',0,np.zeros(len(ev)))
            record('PRIOR',0,np.full(len(ev),labels.mean()))
            for name,own in [('B0_ownlogit',True),('B1_multilogit',False)]:
                f=feature(x,s,own);mu,fs,w=logistic(f[tr],labels);record(name,0,sigmoid(np.c_[np.ones(len(ev)),(f[ev]-mu)/fs]@w));completed+=1
                save(out/f'{year}_{name}_coefficients.json',dict(mu=mu.tolist(),scale=fs.tolist(),weights=w.tolist()))
            for name in MODEL_NAMES:
                for seed in SEEDS:
                    torch.manual_seed(seed);rng=np.random.default_rng(seed);m=build_model(name);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
                    criterion=torch.nn.BCEWithLogitsLoss();best=float('inf');beststate=None;bestepoch=0;start=time.monotonic()
                    for epoch in range(1,21):
                        m.train();order=rng.permutation(tr);total=0.
                        for batch in np.array_split(order,int(np.ceil(len(order)/64))):
                            opt.zero_grad();loss=criterion(m(X[batch],S[batch]),Y[batch])
                            if not torch.isfinite(loss):raise RuntimeError('Nonfinite neural train loss')
                            loss.backward();gradnorm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                            if not torch.isfinite(gradnorm):raise RuntimeError('Nonfinite neural gradient')
                            opt.step();total+=float(loss.detach())*len(batch)
                        m.eval()
                        with torch.no_grad():
                            cal_loss=float(criterion(m(X[cal],S[cal]),Y[cal]));train_loss=total/len(tr)
                        if not np.isfinite(cal_loss):raise RuntimeError('Nonfinite calibration loss')
                        traces.append(dict(year=year,model=name,seed=seed,epoch=epoch,train_loss=train_loss,cal_loss=cal_loss))
                        if cal_loss<best:best=cal_loss;beststate=copy.deepcopy(m.state_dict());bestepoch=epoch
                    m.load_state_dict(beststate);m.eval()
                    with torch.no_grad():p=torch.sigmoid(m(X[ev],S[ev])).numpy()
                    record(name,seed,p);torch.save(beststate,out/f'{year}_{name}_{seed}.pt');completed+=1
                    params.append(dict(year=year,model=name,seed=seed,best_epoch=bestepoch,seconds=time.monotonic()-start,**parameter_counts(m)))
                    pd.DataFrame(traces).to_csv(out/'training_trace.csv',index=False);pd.DataFrame(params).to_csv(out/'fits.csv',index=False)
                    pd.concat(predictions).to_csv(out/'predictions_seeds.csv',index=False)
                    print(f'FIT {completed}/60 {year} {name} seed={seed} cal={best:.5f} epoch={bestepoch}',flush=True)
        save(out/'status.json',dict(status='COMPLETE',total_fits=completed,neural_fits=54,logistic_fits=6,failures=0))
    except Exception as err:
        save(out/'FAILED.json',dict(error=str(err),completed_fits=completed));raise
    pred=pd.concat(predictions);pred.to_csv(out/'predictions_seeds.csv',index=False)
    seedmetrics=[]
    for (name,seed),part in pred.groupby(['model','seed']):seedmetrics.append(dict(model=name,seed=int(seed),**metrics(part.probability_up,part)))
    pd.DataFrame(seedmetrics).to_csv(out/'metrics_seeds.csv',index=False)
    ens=pred.groupby(['index','date','label_end','direction','change_bp','model','vol_state'],as_index=False).probability_up.mean()
    ens.to_csv(out/'predictions_ensemble.csv',index=False)
    rows=[]
    for name,part in ens.groupby('model'):
        rows.append(dict(model=name,period='ALL',**metrics(part.probability_up,part)))
        for year,q in part.groupby(part.date.str[:4]):rows.append(dict(model=name,period=year,**metrics(q.probability_up,q)))
        for state,q in part.groupby('vol_state'):rows.append(dict(model=name,period=state,**metrics(q.probability_up,q)))
    allrows=pd.DataFrame(rows);allrows.to_csv(out/'metrics_ensemble.csv',index=False)
    print(allrows[allrows.period=='ALL'].to_string(index=False),flush=True)

if __name__=='__main__':
    if not (ROOT/'PROTOCOL.md').is_file():
        raise SystemExit('Historical training requires the authorized local V1 protocol and input.')
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve()
    if not out.is_relative_to(ROOT/'outputs'):raise ValueError('Output outside study')
    out.mkdir(parents=True,exist_ok=False);run(a.source,out)
