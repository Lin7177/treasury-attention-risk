"""Execute the frozen 54-classifier plus 3-PCA study exactly once."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,copy,hashlib,json,platform,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from numerical import metrics,fit_logit,predict_logit
from data_target import load_data,split_data,fit_pca,features
from models_target import build_model
from budget import BudgetLedger
ROOT=Path(__file__).resolve().parent
LINEAR=['F0_Own20Logit','F1_FinancialLogit','F2_PCALogit']
NEURAL=['N0_OwnMaskedFixed','N1_Fixed','N2_UniformCross','N3_ContentCross','N4_SelfTarget']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def version():
    try:
        v=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
        if v.returncode==0:return v.stdout.strip()
    except OSError:pass
    f=ROOT/'SOURCE_VERSION.json';return json.loads(f.read_text())['commit'] if f.exists() else 'unversioned_archive'

def validate_freeze():
    freeze=json.loads((ROOT/'FREEZE_MANIFEST.json').read_text());
    for name,h in freeze['sha256'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Frozen document modified: '+name)
    gate=json.loads((ROOT/'outputs/technical_gate.json').read_text())
    if not gate['passed']:raise RuntimeError('Technical gate not passed')
    for name,h in gate['implementation_sha256'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Implementation changed since technical gate: '+name)
    return gate

def run(source,out):
    gate=validate_freeze();cfg=json.loads((ROOT/'frozen_config.json').read_text())
    out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir();(out/'checkpoints').mkdir()
    ledger=BudgetLedger(ROOT/'outputs/fit_budget.jsonl');active=None
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    try:
        raise RuntimeError("Historical source authorization is local only.")
        d,x,spreads,raw_delta=load_data(source);d.to_csv(out/'sample_ledger.csv',index=False)
        dump(out/'manifest.json',dict(date='2026-11-05',commit=version(),source=str(source.resolve()),source_sha256=sha(source),implementation_sha256=gate['implementation_sha256'],freeze_sha256=sha(ROOT/'FREEZE_MANIFEST.json'),config_sha256=sha(ROOT/'frozen_config.json'),lock_sha256=sha(ROOT/'uv.lock'),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',technical_gate=gate))
        predictions=[];fitrows=[];trace=[];rolemetrics=[];flows=[];cache={}
        def record(name,seed,year,p,ix,role='evaluation'):
            q=d.loc[ix].copy();q['model']=name;q['seed']=seed;q['year']=year;q['role']=role;q['probability_up']=np.asarray(p)
            q.to_csv(out/'predictions'/f'{year}_{name}_{seed}_{role}.csv',index=False)
            if role=='evaluation':predictions.append(q)
            rolemetrics.append(dict(year=year,model=name,seed=seed,role=role,**metrics(p,d.loc[ix])))
        for year in cfg['evaluation_years']:
            tr,va,ev,calall=split_data(d,year);y=(d.loc[tr,'direction'].to_numpy()==1).astype(float)
            xm=x[tr].mean((0,1));xs=x[tr].std((0,1));xs[xs==0]=1
            X=torch.tensor((x-xm)/xs,dtype=torch.float32);Y=torch.tensor(d.direction.eq(1).to_numpy(float),dtype=torch.float32)
            scaler=dict(mean=xm.tolist(),scale=xs.tolist(),train_indices=d.loc[tr,'index'].tolist(),train_dates=d.loc[tr,'date'].tolist(),train_label_ends=d.loc[tr,'label_end'].tolist(),validation_indices=d.loc[calall,'index'].tolist(),eval_indices=d.loc[ev,'index'].tolist())
            dump(out/f'scaler_{year}.json',scaler);cache[year]=(tr,va,ev,calall,X,Y)
            flows.append(dict(year=year,train_nonflat=len(tr),validation_nonflat=len(va),validation_all=len(calall),evaluation_all=len(ev),evaluation_flat=int(d.loc[ev,'direction'].eq(0).sum()),train_label_end=d.loc[tr,'label_end'].max(),validation_start=d.loc[calall,'date'].min(),eval_start=d.loc[ev,'date'].min()))
            pd.DataFrame(flows).to_csv(out/'sample_flow.csv',index=False)
            record('DOWN',0,year,np.zeros(len(ev)),ev);record('PRIOR',0,year,np.full(len(ev),y.mean()),ev)
            for name in LINEAR:
                pca=None
                if name.startswith('F2'):
                    job_key=f'{year}_PCA';ledger.claim(job_key,'pca');active=job_key
                    pca=fit_pca(raw_delta,d,tr);dump(out/f'pca_{year}.json',pca)
                    ledger.finish(active,status='complete',details={'train_unique_days':len(pca['unique_day_indices'])});active=None
                f=features(name,x,spreads,pca);expected=cfg['expected_feature_dimensions'][name[:2]]
                if f.shape!=(len(d),expected):raise AssertionError('Feature count mismatch')
                job_key=f'{year}_{name}_0';ledger.claim(job_key,'linear');active=job_key;start=time.monotonic()
                model=fit_logit(f[tr],y);dump(out/'checkpoints'/f'{active}.json',model)
                for role,ix in [('train',tr),('validation',calall),('evaluation',ev)]:record(name,0,year,predict_logit(model,f[ix]),ix,role)
                fitrows.append(dict(job=active,year=year,model=name,seed=0,kind='linear',parameters=len(model['weights']),selected_epoch=None,iterations=len(model['trace'])-1,seconds=time.monotonic()-start))
                ledger.finish(active,status='complete',details={'iterations':len(model['trace'])-1});active=None
                pd.DataFrame(fitrows).to_csv(out/'fits.csv',index=False)
        # All fixed financial baselines are reported before any new neural model.
        pd.DataFrame(rolemetrics).to_csv(out/'role_metrics.csv',index=False)
        print('All 9 financial Logits and 3 PCA transforms complete; frozen neural matrix starts.',flush=True)
        for year in cfg['evaluation_years']:
            tr,va,ev,calall,X,Y=cache[year]
            for name in NEURAL:
                for seed in cfg['seeds']:
                    job_key=f'{year}_{name}_{seed}';ledger.claim(job_key,'neural');active=job_key;start=time.monotonic()
                    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=build_model(name,seed)
                    assert sum(p.numel() for p in model.parameters())==cfg['expected_parameter_counts'][name[:2]]
                    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
                    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);criterion=torch.nn.BCEWithLogitsLoss()
                    best=float('inf');best_state=None;best_epoch=None
                    for epoch in range(1,cfg['epochs']+1):
                        model.train();order=rng.permutation(tr);online=0.;max_grad=0.
                        for batch in np.array_split(order,int(np.ceil(len(order)/cfg['batch_size']))):
                            opt.zero_grad();loss=criterion(model(X[batch]),Y[batch])
                            if not torch.isfinite(loss):raise ArithmeticError('Nonfinite neural loss')
                            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
                            if not torch.isfinite(grad):raise ArithmeticError('Nonfinite neural gradient')
                            opt.step();online+=float(loss.detach())*len(batch);max_grad=max(max_grad,float(grad))
                        model.eval()
                        with torch.no_grad():fixed_train=float(criterion(model(X[tr]),Y[tr]));valid=float(criterion(model(X[va]),Y[va]))
                        if not np.isfinite([fixed_train,valid]).all():raise ArithmeticError('Nonfinite fixed train/validation loss')
                        trace.append(dict(year=year,model=name,seed=seed,epoch=epoch,online_train_bce=online/len(tr),fixed_train_bce=fixed_train,validation_bce=valid,max_unclipped_grad_norm=max_grad))
                        if valid<best:best=valid;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
                    model.load_state_dict(best_state);model.eval()
                    torch.save(best_state,out/'checkpoints'/f'{active}.pt')
                    for role,ix in [('train',tr),('validation',calall),('evaluation',ev)]:
                        with torch.no_grad():p=torch.sigmoid(model(X[ix])).numpy()
                        record(name,seed,year,p,ix,role)
                    fitrows.append(dict(job=active,year=year,model=name,seed=seed,kind='neural',parameters=sum(p.numel() for p in model.parameters()),selected_epoch=best_epoch,iterations=cfg['epochs'],seconds=time.monotonic()-start))
                    pd.DataFrame(trace).to_csv(out/'training_trace.csv',index=False);pd.DataFrame(fitrows).to_csv(out/'fits.csv',index=False);pd.DataFrame(rolemetrics).to_csv(out/'role_metrics.csv',index=False)
                    pd.concat(predictions).to_csv(out/'predictions_seeds.csv',index=False)
                    ledger.finish(active,status='complete',details={'selected_epoch':best_epoch,'best_validation_bce':best});active=None
                    print(f"FIT {len(fitrows)}/54 {year} {name} {seed} validation={best:.5f} epoch={best_epoch}",flush=True)
        pred=pd.concat(predictions);pred.to_csv(out/'predictions_seeds.csv',index=False)
        ens=pred.groupby(['index','date','label_end','direction','change_bp','year','model'],as_index=False).probability_up.mean();ens.to_csv(out/'predictions_ensemble.csv',index=False)
        (out/'budget.jsonl').write_bytes((ROOT/'outputs/fit_budget.jsonl').read_bytes())
        summary=ledger.summary();assert (summary['neural'],summary['linear'],summary['pca'],summary['failed'],summary['pending'])==(45,9,3,0,0);dump(out/'budget_summary.json',summary)
        dump(out/'status.json',dict(status='COMPLETE',classifier_fits=54,pca_fits=3,failures=0,budget=summary))
    except Exception as e:
        for key,filename in [('trace','training_trace.csv'),('fitrows','fits.csv'),('rolemetrics','role_metrics.csv')]:
            if key in locals():pd.DataFrame(locals()[key]).to_csv(out/filename,index=False)
        if active is not None:
            try:ledger.finish(active,status='failed',details={'error':str(e)})
            except Exception:pass
        if (ROOT/'outputs/fit_budget.jsonl').exists():(out/'budget.jsonl').write_bytes((ROOT/'outputs/fit_budget.jsonl').read_bytes())
        dump(out/'FAILED.json',dict(error=str(e),active_job=active,budget=ledger.summary()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();o=a.output.resolve()
    if not o.is_relative_to(ROOT/'outputs'):raise ValueError('Output outside new stage')
    run(a.source,o)
