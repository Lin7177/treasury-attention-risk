"""Frozen paired analysis for Target-Centric v1. No fitting occurs here."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
YEARS=('2022','2023','2024')
SEED=20261104
REPETITIONS=2000
PAIR_NAMES=(
 ('N3_ContentCross','N2_UniformCross'),
 ('N3_ContentCross','N1_Fixed'),
 ('F1_FinancialLogit','F0_Own20Logit'),
 ('F2_PCALogit','F0_Own20Logit'),
 ('N1_Fixed','N0_OwnMaskedFixed'),
 ('N0_OwnMaskedFixed','F0_Own20Logit'),
 ('N3_ContentCross','N4_SelfTarget'),
 ('N4_SelfTarget','OLD_T1_mean'))
PRIMARY_PAIRS=PAIR_NAMES[:2]
FROZEN_MODELS=('F0_Own20Logit','F1_FinancialLogit','F2_PCALogit','N0_OwnMaskedFixed','N1_Fixed','N2_UniformCross','N3_ContentCross','N4_SelfTarget')


def metrics(probability,direction):
    p=np.asarray(probability,dtype=float);y=np.asarray(direction,dtype=int)
    if len(p)!=len(y):raise ValueError('Probability and target lengths differ')
    if p.ndim!=1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid probabilities')
    pred=np.where(p>=.5,1,-1);nf=y!=0
    up=float((pred[y==1]==1).mean()) if np.any(y==1) else np.nan
    down=float((pred[y==-1]==-1).mean()) if np.any(y==-1) else np.nan
    pn=np.clip(p[nf],1e-12,1-1e-12);yn=(y[nf]==1).astype(float)
    return dict(n=len(y),nonflat_n=int(nf.sum()),flat_n=int(np.sum(y==0)),
        accuracy=float((pred==y).mean()) if len(y) else np.nan,
        balanced_accuracy=(up+down)/2,up_recall=up,down_recall=down,
        predicted_up_share=float((pred==1).mean()) if len(y) else np.nan,
        true_up_share=float((y==1).mean()) if len(y) else np.nan,
        true_down_share=float((y==-1).mean()) if len(y) else np.nan,
        brier=float(np.mean((p[nf]-yn)**2)) if nf.any() else np.nan,
        log_loss=float(-np.mean(yn*np.log(pn)+(1-yn)*np.log1p(-pn))) if nf.any() else np.nan,
        probability_q05=float(np.quantile(p,.05)) if len(p) else np.nan,
        probability_q50=float(np.quantile(p,.5)) if len(p) else np.nan,
        probability_q95=float(np.quantile(p,.95)) if len(p) else np.nan)


def macro_ba(probability,direction,years,expected_years=YEARS):
    years=np.asarray(years,dtype=str);values=[]
    for year in expected_years:
        mask=years==str(year)
        if not mask.any():return np.nan
        values.append(metrics(np.asarray(probability)[mask],np.asarray(direction)[mask])['balanced_accuracy'])
    return float(np.mean(values)) if np.isfinite(values).all() else np.nan


def joint_block_indices(years,length,rng):
    """One shared draw for all model columns/pairs; never bridge annual folds."""
    years=np.asarray(years,dtype=str);pieces=[]
    for year in dict.fromkeys(years.tolist()):
        ids=np.flatnonzero(years==year)
        size=min(int(length),len(ids))
        starts=rng.integers(0,len(ids)-size+1,size=int(np.ceil(len(ids)/size)))
        pieces.append(np.concatenate([ids[s:s+size] for s in starts])[:len(ids)])
    return np.concatenate(pieces) if pieces else np.array([],dtype=int)


def matrix_statistics(probabilities,direction,years,expected_years=YEARS):
    """Simultaneously evaluate all model columns on exactly the same draw."""
    p=np.asarray(probabilities,float);y=np.asarray(direction,int);years=np.asarray(years,str)
    if p.ndim!=2 or len(p)!=len(y):raise ValueError('Expected aligned N by model probability matrix')
    pred=np.where(p>=.5,1,-1)
    def ba(mask):
        up=mask&(y==1);down=mask&(y==-1)
        if not up.any() or not down.any():return np.full(p.shape[1],np.nan)
        return ((pred[up]==1).mean(0)+(pred[down]==-1).mean(0))/2
    yearly=[ba(years==str(year)) for year in expected_years]
    nf=y!=0
    result={'macro_balanced_accuracy':np.mean(yearly,axis=0),
            'pooled_balanced_accuracy':ba(np.ones(len(y),bool)),
            'brier':np.mean((p[nf]-(y[nf]==1)[:,None])**2,axis=0) if nf.any() else np.full(p.shape[1],np.nan)}
    result.update({f'{year}_balanced_accuracy':value for year,value in zip(expected_years,yearly)})
    return result


def paired_bootstrap(probabilities,direction,years,names,pairs=PAIR_NAMES,repetitions=REPETITIONS,block_lengths=(5,10,20),seed=SEED,minimum_valid=1900,expected_years=YEARS):
    names=list(names);lookup={name:i for i,name in enumerate(names)}
    for left,right in pairs:
        if left not in lookup or right not in lookup:raise ValueError('Pair absent from model matrix')
    observed=matrix_statistics(probabilities,direction,years,expected_years)
    rows=[]
    for length in block_lengths:
        rng=np.random.default_rng(seed)
        draws={metric:[] for metric in observed}
        for _ in range(repetitions):
            ids=joint_block_indices(years,length,rng)
            value=matrix_statistics(np.asarray(probabilities)[ids],np.asarray(direction)[ids],np.asarray(years)[ids],expected_years)
            for metric,array in value.items():draws[metric].append(array)
        for left,right in pairs:
            li,ri=lookup[left],lookup[right]
            for metric,values in draws.items():
                delta=np.asarray(values)[:,li]-np.asarray(values)[:,ri]
                valid=delta[np.isfinite(delta)]
                levels=[.95,.975] if (left,right) in PRIMARY_PAIRS and metric=='macro_balanced_accuracy' else [.95]
                for confidence in levels:
                    alpha=(1-confidence)/2
                    bounds=np.quantile(valid,[alpha,1-alpha]) if len(valid) else [np.nan,np.nan]
                    rows.append(dict(left=left,right=right,metric=metric,block_length=int(length),confidence_level=confidence,
                        point_estimate=float(observed[metric][li]-observed[metric][ri]),low=bounds[0],high=bounds[1],
                        valid_replicates=len(valid),requested_replicates=repetitions,minimum_valid=minimum_valid,
                        valid_enough=len(valid)>=minimum_valid,bootstrap_seed=seed))
    return pd.DataFrame(rows)

def validate_prediction_frame(frame,expected_models=None):
    required={'index','date','label_end','direction','model','probability_up'}
    if not required.issubset(frame):raise ValueError('Prediction columns missing: '+str(required-set(frame)))
    if expected_models is not None and not set(expected_models).issubset(frame.model.unique()):raise ValueError('Frozen model predictions missing')
    if not frame.direction.isin([-1,0,1]).all():raise ValueError('Invalid direction labels')
    if not np.isfinite(frame.probability_up).all() or not frame.probability_up.between(0,1).all():raise ValueError('Invalid probabilities')
    if frame.duplicated(['model','index']).any() or frame.duplicated(['model','date']).any():raise ValueError('Duplicate prediction keys')
    reference=None
    for name,part in frame.groupby('model'):
        part=part.sort_values('index').reset_index(drop=True)
        if len(part)!=586 or part.direction.ne(0).sum()!=574:raise ValueError(f'{name}: sample-flow mismatch')
        if part.date.max()>'2024-06-21' or part.label_end.max()>'2024-06-28':raise ValueError('Outside frozen evaluation range')
        if not (part.label_end>part.date).all():raise ValueError('Invalid label maturity endpoints')
        for year,n,flat in [('2022',237,8),('2023',237,2),('2024',112,2)]:
            q=part[part.date.str.startswith(year)]
            if len(q)!=n or q.direction.eq(0).sum()!=flat:raise ValueError(f'{name}/{year}: annual sample-flow mismatch')
            if (q.label_end>f'{year}-12-31').any():raise ValueError('Annual target label crosses frozen year boundary')
        identity=part[['index','date','label_end','direction']]
        if reference is None:reference=identity
        elif not identity.equals(reference):raise ValueError(f'{name}: cross-model target keys differ')
    return reference


def analyze(run,old_path,output):
    import hashlib
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    ensemble=pd.read_csv(run/'predictions_ensemble.csv')
    validate_prediction_frame(ensemble,FROZEN_MODELS)
    unexpected=set(ensemble.model.unique())-set(FROZEN_MODELS)-{'DOWN','PRIOR'}
    if unexpected:raise ValueError(f'Unregistered models: {unexpected}')
    old=pd.read_csv(old_path);old=old[old.model.eq('T1_attention')].copy();old['model']='OLD_T1_mean'
    validate_prediction_frame(old,['OLD_T1_mean'])
    all_predictions=pd.concat([ensemble,old],ignore_index=True)
    reference=validate_prediction_frame(all_predictions,FROZEN_MODELS+('OLD_T1_mean',))
    names=[name for name in FROZEN_MODELS]+[name for name in ['DOWN','PRIOR'] if name in ensemble.model.unique()]+['OLD_T1_mean']
    arrays=[];models=[];annual=[]
    for name in names:
        part=all_predictions[all_predictions.model.eq(name)].sort_values('index')
        p=part.probability_up.to_numpy();y=part.direction.to_numpy();years=part.date.str[:4].to_numpy()
        arrays.append(p)
        models.append(dict(model=name,macro_balanced_accuracy=macro_ba(p,y,years),**metrics(p,y)))
        for year in YEARS:
            q=part[part.date.str.startswith(year)]
            annual.append(dict(model=name,year=year,**metrics(q.probability_up,q.direction)))
    model_table=pd.DataFrame(models);year_table=pd.DataFrame(annual)
    matrix=np.column_stack(arrays);direction=reference.direction.to_numpy();years=reference.date.str[:4].to_numpy()
    intervals=paired_bootstrap(matrix,direction,years,names)
    seedframe=pd.read_csv(run/'predictions_seeds.csv')
    seed_rows=[]
    if 'seed' not in seedframe:raise ValueError('Per-seed file missing seed column')
    for (name,seed),part in seedframe.groupby(['model','seed']):
        check=part.copy();check['model']=f'{name}|{seed}';seed_identity=validate_prediction_frame(check)
        if not seed_identity.equals(reference):raise ValueError('Per-seed targets differ from ensemble targets')
        part=part.sort_values('index');p=part.probability_up.to_numpy();y=part.direction.to_numpy();syears=part.date.str[:4].to_numpy()
        seed_rows.append(dict(model=name,seed=int(seed),period='ALL',macro_balanced_accuracy=macro_ba(p,y,syears),**metrics(p,y)))
        for year in YEARS:
            q=part[part.date.str.startswith(year)]
            seed_rows.append(dict(model=name,seed=int(seed),period=year,macro_balanced_accuracy=np.nan,**metrics(q.probability_up,q.direction)))
    yearly_differences=[]
    indexed=year_table.set_index(['model','year'])
    for left,right in PAIR_NAMES:
        for year in YEARS:
            a=indexed.loc[(left,year)];b=indexed.loc[(right,year)]
            yearly_differences.append(dict(left=left,right=right,year=year,n=int(a.n),
                balanced_accuracy_delta=a.balanced_accuracy-b.balanced_accuracy,
                accuracy_delta=a.accuracy-b.accuracy,brier_delta=a.brier-b.brier))
    yearly_differences=pd.DataFrame(yearly_differences)
    continuation=[];indexed_models=model_table.set_index('model')
    for left,right in PRIMARY_PAIRS:
        q=intervals[(intervals.left==left)&(intervals.right==right)&intervals.metric.eq('macro_balanced_accuracy')&(intervals.block_length==10)&(intervals.confidence_level==.975)].iloc[0]
        yd=yearly_differences[(yearly_differences.left==left)&(yearly_differences.right==right)]
        brier_delta=float(indexed_models.loc[left,'brier']-indexed_models.loc[right,'brier'])
        criteria=dict(macro_point_positive=bool(q.point_estimate>0),conservative_interval_lower_positive=bool(np.isfinite(q.low) and q.low>0),
            at_least_two_years_positive=bool(yd.balanced_accuracy_delta.gt(0).sum()>=2),pooled_brier_not_worse=bool(brier_delta<=0),valid_replicates_at_least_1900=bool(q.valid_replicates>=1900))
        continuation.append(dict(left=left,right=right,macro_point=float(q.point_estimate),macro_975_low=float(q.low) if np.isfinite(q.low) else None,
            macro_975_high=float(q.high) if np.isfinite(q.high) else None,positive_years=int(yd.balanced_accuracy_delta.gt(0).sum()),
            pooled_brier_delta=brier_delta,valid_replicates=int(q.valid_replicates),criteria=criteria,passes_all=all(criteria.values())))
    decision=dict(stage='target_centric_v1',exploratory_continuation_signal=all(item['passes_all'] for item in continuation),
        comparisons=continuation,decision_rule='Both predeclared primary comparisons must pass all five frozen criteria; no new search or fits authorized by this flag.',
        evidence_scope='Exposed historical development only; intervals conditional on trained ensembles, not independent confirmation or training reproduction.',
        bootstrap=dict(seed=SEED,repetitions=REPETITIONS,block_lengths=[5,10,20],joint_model_draws=True,missing_year_class='undefined_macro_no_redraw'),
        financial_requests=586,nonflat_requests=574,analysis_sha256=digest(__file__),ensemble_sha256=digest(run/'predictions_ensemble.csv'),
        seed_predictions_sha256=digest(run/'predictions_seeds.csv'),old_t1_file_sha256=digest(old_path),
        frozen_protocol_sha256=digest(ROOT/'FROZEN_PROTOCOL.md'),frozen_config_sha256=digest(ROOT/'frozen_config.json'))
    targets={'model_metrics.csv':model_table,'year_metrics.csv':year_table,'seed_metrics.csv':pd.DataFrame(seed_rows),
             'paired_intervals.csv':intervals,'year_pair_deltas.csv':yearly_differences}
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'outputs'):raise ValueError('Analysis output must stay in new-stage outputs')
    for name in list(targets)+['continuation_decision.json']:
        if (output/name).exists():raise FileExistsError(f'Preserve prior analysis: {output/name}')
    output.mkdir(parents=True,exist_ok=True)
    for name,table in targets.items():table.to_csv(output/name,index=False)
    (output/'continuation_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(model_table[['model','n','accuracy','balanced_accuracy','macro_balanced_accuracy','brier']].to_string(index=False))
    print(json.dumps(decision,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'outputs'/'target_main_001')
    parser.add_argument('--old',type=Path,default=ROOT.parent/'outputs'/'main_001'/'predictions_ensemble.csv')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();analyze(args.run,args.old,args.output or args.run)


if __name__=='__main__':main()
