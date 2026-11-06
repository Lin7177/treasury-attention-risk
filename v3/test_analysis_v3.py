"""Synthetic-only guards for the frozen V3 analysis. No market fitting."""
import json
from datetime import timedelta
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_v3 import (
    MODELS, PAIRS, YEARS, COUNTS, SEEDS, qlike, paired_difference,
    matrix_statistics, moving_block_indices, joint_bootstrap, interval_table,
    align_ensemble, validate_seeds, tail_sensitivity, continuation_decision,
    describe_models, evaluation_rows, run)

def fixture():
    records=[];index=0
    for year in YEARS:
        dates=pd.bdate_range(str(year)+"-01-03", periods=COUNTS[year])
        for date in dates:
            q=0. if index%17==0 else float(1+index%8)
            for j,name in enumerate(MODELS):
                records.append(dict(year=year,index=index,date=date.strftime("%Y-%m-%d"),
                    information_end=(date-timedelta(days=1)).strftime("%Y-%m-%d"),
                    label_end=(date+timedelta(days=7)).strftime("%Y-%m-%d"),
                    q5=q,model=name,forecast=2.+j*.2+index*.001))
            index+=1
    return pd.DataFrame(records)

class AnalysisTests(unittest.TestCase):
    def test_zero_target_retained_and_invalid_forecast_rejected(self):
        np.testing.assert_allclose(qlike([0.,2.],[2.,2.]),[np.log(2),np.log(2)+1])
        for bad in ([0.,1.],[-1.,1.],[float("nan"),1.]):
            with self.assertRaises(ValueError):qlike([0.,2.],bad)
    def test_units_pair_and_intervals_invariant(self):
        y=np.repeat([2022,2023,2024],[30,35,25])
        q=np.arange(90,dtype=float)%7
        a=np.linspace(.3,3,90);b=np.linspace(2,.2,90)
        d=paired_difference(q,a,b)
        np.testing.assert_allclose(d,paired_difference(q*1e8,a*1e8,b*1e8),atol=1e-13)
        np.testing.assert_allclose(qlike(q*1e8,a*1e8)-qlike(q,a),np.log(1e8),atol=1e-13)
        x=joint_bootstrap(d,y,10,31)
        z=joint_bootstrap(paired_difference(q*1e8,a*1e8,b*1e8),y,10,31)
        for key in x:np.testing.assert_allclose(x[key],z[key],atol=1e-13)
    def test_macro_not_pooled(self):
        v=np.array([0.,0.,0.,9.,3.]);y=[2022,2022,2022,2023,2024]
        out=matrix_statistics(v,y)
        self.assertEqual(out["macro"][0],4.)
        self.assertEqual(out["pooled"][0],2.4)
    def test_joint_sampling_columns_and_no_crossyear(self):
        v=np.arange(90,dtype=float);y=np.repeat([2022,2023,2024],[30,35,25])
        out=joint_bootstrap(np.c_[v,v,2*v,np.zeros(90)],y,10,31)
        for a in out.values():
            np.testing.assert_array_equal(a[:,0],a[:,1])
            np.testing.assert_allclose(a[:,2],2*a[:,0],atol=1e-14)
            np.testing.assert_array_equal(a[:,3],np.zeros(31))
        # Manual first replicate exactly matches sequential annual RNG draws.
        rng=np.random.default_rng(20261105);means=[]
        for year in [2022,2023,2024]:
            sub=v[y==year];means.append(sub[moving_block_indices(len(sub),10,rng)].mean())
        np.testing.assert_allclose(out["macro"][0,:3],[np.mean(means),np.mean(means),2*np.mean(means)])
    def test_non_circular_blocks_and_reproducibility(self):
        class Ends:
            def integers(self,low,high,size):
                self.bounds=(low,high,size)
                return np.full(size,high-1)
        rng=Ends();a=moving_block_indices(23,10,rng)
        self.assertEqual(rng.bounds,(0,14,3))
        np.testing.assert_array_equal(a,np.tile(np.arange(13,23),3)[:23])
        self.assertTrue((a<23).all())
    def test_zero_difference_ci_all_zero(self):
        v=np.zeros((90,6));y=np.repeat([2022,2023,2024],30)
        out=interval_table(v,y,blocks=(10,),replications=30)
        self.assertEqual(len(out),32)
        self.assertTrue((out[["estimate","lower","upper"]]==0).all().all())
        self.assertEqual(len(out[out.confidence==.975]),2)
    def test_exact_keys_counts_order_and_missing(self):
        f=fixture().sample(frac=1,random_state=4)
        ref,p=align_ensemble(f)
        self.assertEqual(len(ref),589);self.assertEqual(p.shape,(589,8))
        with self.assertRaises(ValueError):align_ensemble(f.iloc[:-1])
        with self.assertRaises(ValueError):align_ensemble(pd.concat([f,f.iloc[:1]]))
        wrong=f.copy();i=wrong.index[wrong.model=="Fixed"][0];wrong.loc[i,"q5"]+=1
        with self.assertRaises(ValueError):align_ensemble(wrong)
    def test_reject_mixed_roles_future_and_bad_indices(self):
        f=fixture();f["role"]="evaluation";f.loc[0,"role"]="train"
        with self.assertRaises(ValueError):evaluation_rows(f)
        f=fixture();f.loc[0,"label_end"]="2024-07-01"
        with self.assertRaises(ValueError):evaluation_rows(f)
        f=fixture();f["index"]=f["index"].astype(float);f.loc[0,"index"]=float("inf")
        with self.assertRaises(ValueError):evaluation_rows(f)
    def test_seed_average_is_physical_arithmetic_and_complete(self):
        f=fixture();ref,p=align_ensemble(f);pieces=[]
        for name in MODELS:
            row=f[f.model==name]
            for seed,shift in zip(SEEDS if name in ("Fixed","Uniform","Content") else (0,),
                                   (-.2,0.,.2) if name in ("Fixed","Uniform","Content") else (0.,)):
                item=row.copy();item["seed"]=seed;item["forecast"]+=shift
                item["role"]="evaluation";pieces.append(item)
        seeds=pd.concat(pieces,ignore_index=True)
        report=validate_seeds(seeds,ref,p)
        self.assertEqual(len(report),56)
        bad=seeds.copy();bad.loc[bad.index[0],"forecast"]+=.1
        with self.assertRaises(AssertionError):validate_seeds(bad,ref,p)
    def test_tail_count_ties_and_no_main_exclusion(self):
        f=fixture();ref,p=align_ensemble(f);ref.q5=0.
        tail=tail_sensitivity(ref,p)
        self.assertEqual(len(tail),18)
        self.assertEqual(tail.groupby("year").n_tail.first().to_dict(),{2022:3,2023:3,2024:2})
        self.assertTrue(tail.q_share.isna().all())
        first=ref[ref.year==2022].date.iloc[0]
        self.assertTrue((tail[tail.year==2022].largest_q_date==first).all())
        self.assertEqual(len(ref),589)
    def test_primary_flag_not_financial_advantage(self):
        f=fixture();ref,p=align_ensemble(f)
        models,_=describe_models(ref,p)
        intervals=pd.DataFrame([dict(model=m,baseline=b,statistic="macro",block=10,
            confidence=.975,estimate=-.1,upper=-.01) for m,b in PAIRS[:2]])
        annual=pd.DataFrame([dict(model=m,baseline=b,year=y,delta=-.1 if y!=2024 else .01)
            for m,b in PAIRS[:2] for y in YEARS])
        d=continuation_decision(intervals,annual,models)
        self.assertTrue(d["mechanism_signal"])
        self.assertFalse(d["financial_significance_tested"])
        self.assertEqual(len(d["content_minus_financial_macro_point_differences"]),5)
        intervals.loc[0,"upper"]=0.
        self.assertFalse(continuation_decision(intervals,annual,models)["mechanism_signal"])
    def test_run_requires_complete_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);output=path/"out"
            with self.assertRaises(ValueError):run(path,output)
            output.mkdir()
            with self.assertRaises(FileExistsError):run(path,output)

    def test_complete_synthetic_pipeline_no_market(self):
        f=fixture();parts=[]
        for name in MODELS:
            row=f[f.model==name]
            for seed in SEEDS if name in ("Fixed","Uniform","Content") else (0,):
                part=row.copy();part["seed"]=seed;part["role"]="evaluation"
                parts.append(part)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);out=path/"report"
            f.to_csv(path/"predictions_ensemble.csv",index=False)
            pd.concat(parts).to_csv(path/"predictions_seeds.csv",index=False)
            (path/"status.json").write_text('{"status":"COMPLETE"}')
            result=run(path,out)
            self.assertIn("mechanism_signal",result)
            self.assertEqual(len(pd.read_csv(out/"paired_intervals.csv")),96)
            self.assertEqual(len(pd.read_csv(out/"year_pair_deltas.csv")),18)
            self.assertEqual(len(pd.read_csv(out/"model_metrics.csv")),16)
            manifest=json.loads((out/"analysis_manifest.json").read_text())
            self.assertEqual(manifest["real_fits"],0)
            self.assertEqual(manifest["eval_requests"],589)
            with self.assertRaises(FileExistsError):run(path,out)

if __name__=="__main__":
    unittest.main()
