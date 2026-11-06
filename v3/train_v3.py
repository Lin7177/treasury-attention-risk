"""Single bounded execution of the peer-reviewed V3 Q5 study; no retries."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,copy,hashlib,json,platform,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from budget import BudgetLedger
from numerical_v3 import fit_har,predict_har,har_design,metrics,FLOOR,positive_guard
from data_v3 import load_data,split_data,fit_input_transform,transform_x,verify_frozen_counts
from models_v3 import build_model
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def version():
    try:
        s=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
        if s.returncode==0:return s.stdout.strip()
    except OSError:pass
    return json.loads((ROOT/'SOURCE_VERSION.json').read_text())['training_commit'] if (ROOT/'SOURCE_VERSION.json').exists() else 'archive_unversioned'
def validate():
    freeze=json.loads((ROOT/'FREEZE_MANIFEST.json').read_text())
    for name,h in freeze['sha256'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Frozen material changed: '+name)
    gate=json.loads((ROOT/'outputs/technical_gate.json').read_text())
    if not gate['passed']:raise RuntimeError('Technical gate not passed')
    for name,h in gate['implementation_sha256'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Runtime changed after gate: '+name)
    return gate,json.loads((ROOT/'frozen_config.json').read_text())

def run(source,out):
    gate,cfg=validate()
    if sha(source)!=cfg['source_sha256']:raise RuntimeError('Authorized input identity mismatch')
    out.mkdir(parents=True,exist_ok=False)
    for name in ['checkpoints','predictions']:(out/name).mkdir()
    journal=ROOT/'outputs/fit_budget.jsonl';ledger=BudgetLedger(journal);active=None
    predictions=[];fitrows=[];trace=[];roles=[];flows=[];cache={};initializations=[];shared_init={};attention_init={}
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    def persist():
        for name,rows in [('fits.csv',fitrows),('training_trace.csv',trace),('role_metrics.csv',roles),('sample_flow.csv',flows)]:pd.DataFrame(rows).to_csv(out/name,index=False)
        if predictions:pd.concat(predictions).to_csv(out/'predictions_seeds.csv',index=False)
    try:
        d,x,raw_delta=load_data(source);verify_frozen_counts(d);d.to_csv(out/'sample_ledger.csv',index=False);save(out/'data_flow.json',d.attrs['flow'])
        save(out/'manifest.json',dict(date='2026-11-05',commit=version(),source=str(source.resolve()),source_sha256=sha(source),freeze_sha256=sha(ROOT/'FREEZE_MANIFEST.json'),config_sha256=sha(ROOT/'frozen_config.json'),lock_sha256=sha(ROOT/'uv.lock'),implementation_sha256=gate['implementation_sha256'],python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,device='cpu',new_optimizer_limit=33,evidence='Exposed historical retrospective evaluation, not independent confirmation',technical_gate=gate))
        def record(name,seed,year,ix,forecast,role):
            q=d.iloc[ix].copy();f=np.asarray(forecast,dtype=float);q['forecast']=f;q['model']=name;q['seed']=seed;q['year']=year;q['role']=role
            metric=metrics(q.q5.to_numpy(),f);q.to_csv(out/'predictions'/f'{year}_{name}_{seed}_{role}.csv',index=False)
            roles.append(dict(year=year,model=name,seed=seed,role=role,**metric))
            if role=='evaluation':predictions.append(q)
        for year in cfg['evaluation_years']:
            tr,va,ev=split_data(d,year);s=float(d.iloc[tr].q5.mean());assert np.isfinite(s) and s>0
            transform=fit_input_transform(raw_delta,d,tr);X=torch.tensor(transform_x(x,transform),dtype=torch.float32);Y=torch.tensor(d.q5.to_numpy()/s,dtype=torch.float32)
            save(out/f'transform_{year}.json',dict(**transform,target_scale=s,train_positions=tr.tolist(),validation_positions=va.tolist(),evaluation_positions=ev.tolist(),train_source_indices=d.iloc[tr]['index'].tolist(),training_statistics_not_optimizer_fits=True))
            flows.append(dict(year=year,train=len(tr),validation=len(va),evaluation=len(ev),target_scale=s,train_zero=int(d.iloc[tr].q5.eq(0).sum()),eval_zero=int(d.iloc[ev].q5.eq(0).sum()),train_label_end=d.iloc[tr].label_end.max(),validation_start=d.iloc[va].date.min(),eval_start=d.iloc[ev].date.min(),unique_input_days=len(transform['unique_day_indices'])))
            cache[year]=(tr,va,ev,X,Y,s)
            w=(1-.94)*.94**np.arange(19,-1,-1)/(1-.94**20);assert abs(w.sum()-1)<1e-12
            baseline={'Mean':np.full(len(d),s),'SMA20':5*np.square(x[:,:,3]).mean(1),'EWMA':5*(np.square(x[:,:,3])*w).sum(1)}
            counts=[]
            for name,f in baseline.items():
                floor_hits=int((f[ev]<s*FLOOR).sum());counts.append(dict(model=name,evaluation_floor_hits=floor_hits));f=np.maximum(f,s*FLOOR)
                record(name,0,year,ev,f[ev],'evaluation')
            save(out/f'baseline_floor_{year}.json',counts)
            for name in cfg['har_models']:
                design=har_design(x,name=='OwnHAR',s);job=f'{year}_{name}_0';ledger.claim(job,'har');active=job;start=time.monotonic()
                state=fit_har(design[tr],d.iloc[tr].q5.to_numpy());assert state['target_scale']==s
                save(out/'checkpoints'/f'{job}.json',state)
                for role,ix in [('train',tr),('validation',va),('evaluation',ev)]:record(name,0,year,ix,predict_har(state,design[ix],role),role)
                fitrows.append(dict(job=job,kind='har',year=year,model=name,seed=0,parameters=len(state['beta']),selected_epoch=None,iterations=len(state['trace'])-1,seconds=time.monotonic()-start))
                ledger.finish(job,'complete',{'gradient_inf':state['trace'][-1]['gradient_inf']});active=None;persist()
        print('6 financial HAR fits complete. Frozen 27-neural matrix starts.',flush=True)
        for year in cfg['evaluation_years']:
            tr,va,ev,X,Y,s=cache[year]
            for name in cfg['neural_models']:
                for seed in cfg['seeds']:
                    job=f'{year}_{name}_{seed}';ledger.claim(job,'neural');active=job;start=time.monotonic();torch.manual_seed(seed);rng=np.random.default_rng(seed);model=build_model(name,seed)
                    assert sum(v.numel() for v in model.parameters())==cfg['parameter_counts'][name]
                    init_group=model.initialization_hashes()
                    if seed in shared_init:assert shared_init[seed]==init_group['shared']
                    else:shared_init[seed]=init_group['shared']
                    if name in ['Uniform','Content']:
                        if seed in attention_init:assert attention_init[seed]==init_group['mixer']
                        else:attention_init[seed]=init_group['mixer']
                    initializations.append(dict(year=year,model=name,seed=seed,**init_group));save(out/'initialization_hashes.json',initializations)
                    init_hash=hashlib.sha256(b''.join(k.encode()+v.detach().numpy().tobytes() for k,v in model.state_dict().items())).hexdigest()
                    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);best=float('inf');best_state=None;best_epoch=None
                    def model_u(ix,role):
                        u,raw=model(X[ix],return_raw=True)
                        positive_guard(raw.detach().numpy(),role,dict(year=year,model=name,seed=seed))
                        if not torch.isfinite(u).all():raise ArithmeticError('Nonfinite positive neural forecast')
                        return u
                    def loss_for(ix,role):
                        u=model_u(ix,role);return (torch.log(u)+Y[ix]/u).mean()
                    for epoch in range(1,cfg['epochs']+1):
                        model.train();order=rng.permutation(tr);online=0.;maxgrad=0.
                        for ids in np.array_split(order,int(np.ceil(len(order)/cfg['batch_size']))):
                            opt.zero_grad();loss=loss_for(ids,f'train_batch/epoch{epoch}')
                            if not torch.isfinite(loss):raise ArithmeticError('Nonfinite neural QLIKE')
                            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
                            if not torch.isfinite(grad):raise ArithmeticError('Nonfinite gradient')
                            opt.step();online+=float(loss.detach())*len(ids);maxgrad=max(maxgrad,float(grad))
                        model.eval()
                        with torch.no_grad():train_fixed=float(loss_for(tr,'train'));valid=float(loss_for(va,'validation'))
                        if not np.isfinite([train_fixed,valid]).all():raise ArithmeticError('Nonfinite fixed role loss')
                        trace.append(dict(year=year,model=name,seed=seed,epoch=epoch,online_normalized_qlike=online/len(tr),fixed_train_normalized_qlike=train_fixed,validation_normalized_qlike=valid,max_unclipped_grad=maxgrad))
                        if valid<best:best=valid;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
                        pd.DataFrame(trace).to_csv(out/'training_trace.csv',index=False)
                    model.load_state_dict(best_state);model.eval();torch.save(best_state,out/'checkpoints'/f'{job}.pt')
                    for role,ix in [('train',tr),('validation',va),('evaluation',ev)]:
                        with torch.no_grad():f=model_u(ix,role).numpy().astype(float)*s
                        record(name,seed,year,ix,f,role)
                    fitrows.append(dict(job=job,kind='neural',year=year,model=name,seed=seed,parameters=sum(v.numel() for v in model.parameters()),selected_epoch=best_epoch,iterations=cfg['epochs'],seconds=time.monotonic()-start,initial_state_sha256=init_hash))
                    ledger.finish(job,'complete',{'selected_epoch':best_epoch,'validation_normalized_qlike':best});active=None;persist();print(f'FIT {len(fitrows)}/33 {year} {name} {seed} validation={best:.6f} epoch={best_epoch}',flush=True)
        pred=pd.concat(predictions);keys=['index','date','information_end','label_end','q5','calendar_days','year','model'];ensemble=pred.groupby(keys,as_index=False).forecast.mean();ensemble.to_csv(out/'predictions_ensemble.csv',index=False)
        summary=ledger.summary();assert (summary['neural'],summary['har'],summary['pca'],summary['failed'],summary['pending'])==(27,6,0,0,0)
        (out/'budget.jsonl').write_bytes(journal.read_bytes());save(out/'budget_summary.json',summary);save(out/'status.json',dict(status='COMPLETE',supervised_optimizations=33,neural=27,har=6,failures=0,statistics_estimates=3,source_statistics='Three training MeanQ scales, three unique-day 10channel transforms, six HAR scalers; not additional optimizers',budget=summary))
    except BaseException as e:
        persist()
        if hasattr(e,'trace'):
            pd.DataFrame(e.trace).to_csv(out/'failed_har_trace.csv',index=False)
            safe_trace=[{k:(v if not isinstance(v,float) or np.isfinite(v) else None) for k,v in row.items()} for row in e.trace]
            save(out/'failed_har_trace.json',safe_trace)
        if hasattr(e,'diagnostics'):save(out/'failed_forecast_diagnostics.json',e.diagnostics)
        if active is not None:
            try:ledger.finish(active,'failed',{'error':str(e)})
            except Exception:pass
        if journal.exists():(out/'budget.jsonl').write_bytes(journal.read_bytes())
        save(out/'FAILED.json',dict(error=str(e),error_type=type(e).__name__,diagnostics=getattr(e,'diagnostics',{}),active_job=active,budget=ledger.summary()));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();o=a.output.resolve()
    if not o.is_relative_to(ROOT/'outputs'):raise ValueError('Output outside V3 stage')
    run(a.source,o)
