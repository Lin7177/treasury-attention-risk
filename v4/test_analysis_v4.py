"""V4 synthetic analysis guards only. No market data, fitting, or downloads."""
import copy,json,tempfile,unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import analyze_v4 as a
from clock_v4 import ScientificClock,spent

def config():
    p=a.ROOT/"frozen_config.json"
    if not p.exists():p=a.ROOT/"frozen_config_DRAFT.json"
    return json.loads(p.read_text(encoding="utf-8"))

def fixture(counts=(30,30,30),zero=False):
    rng=np.random.default_rng(120);rows=[];index=0
    for year,n in zip((2022,2023,2024),counts):
        for date in pd.bdate_range(str(year)+"-01-04",periods=n):
            v=rng.normal(size=(5,3));s=v.T@v if not zero else np.zeros((3,3))
            h=np.array([[6.,.5,.4],[.5,5.,.6],[.4,.6,4.]])+np.eye(3)*(index%7)*.1
            for model in a.MODELS:
                seeds=a.SEEDS if model in a.LEARNED else (0,)
                for seed in seeds:
                    out=h.copy()
                    if model in a.LEARNED:out+=np.eye(3)*({a.SEEDS[0]:-.1,a.SEEDS[1]:0,a.SEEDS[2]:.1}[seed])
                    row=dict(phase="exploration",fold=str(year),role="evaluation",model=model,seed=seed,
                        index=index,date=date.strftime("%Y-%m-%d"),
                        information_end=(date-timedelta(days=1)).strftime("%Y-%m-%d"),
                        label_end=(date+timedelta(days=7)).strftime("%Y-%m-%d"),period=str(year),
                        common_d0=4.,common_d1=5.,common_d2=6.)
                    row.update({c:s[i,j] for c,(i,j) in zip(a.YCOL,a.TRI)})
                    row.update({c:out[i,j] for c,(i,j) in zip(a.FCOL,a.TRI)})
                    rows.append(row)
            index+=1
    return pd.DataFrame(rows)

def mock_run_files(root,cfg):
    run=root/"outputs"/"run";out=root/"outputs"/"report";run.mkdir(parents=True)
    (root/"frozen_config.json").write_text(json.dumps(cfg),encoding="utf-8")
    (root/"FROZEN_PROTOCOL.md").write_text("synthetic protocol only",encoding="utf-8")
    (run/"manifest.json").write_text(json.dumps({"phase":"exploration","config_sha256":a.sha(root/"frozen_config.json")}),encoding="utf-8")
    (run/"status.json").write_text('{"status":"COMPLETE"}',encoding="utf-8")
    fixture().to_csv(run/"predictions_seeds.csv",index=False)
    return run,out

class AnalysisV4Tests(unittest.TestCase):
    def test_matrix_zero_units_domain(self):
        rng=np.random.default_rng(1);v=rng.normal(size=(12,3,3));s=v@v.transpose(0,2,1)
        h=s+.5*np.eye(3);b=s+np.eye(3);unit=np.diag([.1,20.,3.])
        np.testing.assert_allclose(a.matrix_loss(unit@s@unit,unit@h@unit)-a.matrix_loss(s,h),
                                  2*np.log(np.diag(unit)).sum()/3,atol=1e-12)
        np.testing.assert_allclose(a.matrix_loss(unit@s@unit,unit@h@unit)-a.matrix_loss(unit@s@unit,unit@b@unit),
                                  a.matrix_loss(s,h)-a.matrix_loss(s,b),atol=1e-12)
        self.assertTrue(np.isfinite(a.matrix_loss(np.zeros_like(s),h)).all())
        with self.assertRaises(ValueError):a.matrix_loss(-np.eye(3)[None],np.eye(3)[None])
        bad=h.copy();bad[0,0,1]+=10
        with self.assertRaises(ValueError):a.matrix_loss(s,bad)
    def test_physical_arithmetic_ensemble_not_log_or_seed_metrics(self):
        f=fixture();raw,e,ref,pred=a.ensemble(f,enforce_counts=False)
        self.assertEqual(pred.shape,(90,10,3,3))
        np.testing.assert_allclose(pred[:,a.MODELS.index("Content")],pred[:,a.MODELS.index("Prior")],atol=1e-14)
        first=raw[(raw.model=="Content")&(raw["index"]==0)]
        member=a.matrices(first,a.FCOL)
        np.testing.assert_allclose(pred[0,a.MODELS.index("Content")],member.mean(0))
        self.assertGreater(np.abs(np.log(np.diagonal(member,axis1=1,axis2=2)).mean(0)-
                              np.log(np.diagonal(member.mean(0)))).max(),1e-5)
    def test_duplicate_missing_target_commonD_and_seed_keys(self):
        f=fixture()
        with self.assertRaises(ValueError):a.ensemble(pd.concat([f,f.iloc[:1]]),False)
        with self.assertRaises(ValueError):a.ensemble(f.iloc[1:],False)
        for column in ["y00","common_d0"]:
            bad=f.copy();bad.loc[bad.index[(bad.model=="Content")][0],column]+=.01
            with self.assertRaises(ValueError):a.ensemble(bad,False)
        bad=f.copy();bad.loc[0,"seed"]=999
        with self.assertRaises(ValueError):a.ensemble(bad,False)
    def test_maturity_role_and_period_boundaries(self):
        f=fixture();val=f[(f.fold=="2022")&(f["index"]<2)].copy()
        val["role"]="validation";val["date"]="2021-12-30";val["index"]=[r for r in range(len(val))]
        val["information_end"]="2021-12-29";val["label_end"]="2022-01-05"
        # Unique rows per model retain one request; duplicate dates not needed.
        val=val.groupby(["model","seed"],as_index=False).first();val["index"]=999
        with self.assertRaises(ValueError):a.ensemble(pd.concat([f,val]),False)
        wrong=f.copy();wrong.loc[0,"information_end"]=wrong.loc[0,"date"]
        with self.assertRaises(ValueError):a.prepare(wrong)
        wrong=f.copy();wrong.loc[0,"period"]="2020-21"
        with self.assertRaises(ValueError):a.ensemble(wrong,False)
    def test_shared_diagonal_and_hedge_closed_form_cap(self):
        h=np.array([[[1.,5.,0.],[5.,100.,5.],[0.,5.,1.]]])
        s=np.array([[[2.,.3,.2],[.3,4.,.5],[.2,.5,3.]]])
        shared=a.common_matrix(h,np.array([[3.,4.,5.]]))
        np.testing.assert_allclose(np.diagonal(shared,axis1=1,axis2=2),[[3,4,5]])
        risk,beta,info=a.hedge_risk(s,h)
        self.assertEqual(info["cap_hits"],1);np.testing.assert_allclose(np.abs(beta).sum(1),[2.])
        aa=np.array([-beta[0,0],1.,-beta[0,1]])
        self.assertAlmostEqual(risk[0],aa@s[0]@aa)
    def test_period_ratios_not_ratio_of_macro_or_daily_ratios(self):
        p=np.repeat(["2022","2023","2024"],[30,40,20])
        risk=np.ones((90,10));risk[p=="2023"]*=10;risk[p=="2024"]*=100
        risk[p=="2022",a.MODELS.index("Content")]*=.5
        risk[p=="2024",a.MODELS.index("Content")]*=1.5
        stat=a.statistics(np.zeros_like(risk),np.zeros_like(risk),risk,p)
        np.testing.assert_allclose(stat["hedge"]["macro"],0.,atol=1e-15)
        self.assertGreater(stat["hedge"]["pooled"][0],.1)
    def test_joint_bootstrap_no_crossperiod_circular_or_pair_randomness(self):
        p=np.repeat(["2022","2023","2024"],30);v=np.arange(90,dtype=float)
        full=np.tile(v[:,None],(1,10));full[:,a.MODELS.index("Content")]=0
        shape=2*full;risk=np.ones((90,10))
        out=a.bootstrap(full,shape,risk,p,10,reps=31)
        for stat in out["full"]:
            x=out["full"][stat]
            np.testing.assert_array_equal(x[:,0],x[:,1])
            np.testing.assert_allclose(out["shape"][stat],2*x,atol=1e-14)
            np.testing.assert_allclose(out["hedge"][stat],0)
        class Last:
            def integers(self,low,high,size):return np.full(size,high-1)
        np.testing.assert_array_equal(a.block_indices(23,10,Last()),np.tile(np.arange(13,23),3)[:23])
        manual_rng=np.random.default_rng(2026110540);means=[]
        for label in sorted(set(p)):
            sub=v[p==label];means.append(-sub[a.block_indices(30,10,manual_rng)].mean())
        self.assertAlmostEqual(out["full"]["macro"][0,0],np.mean(means))
    def test_undefined_bootstrap_missing_and_no_refill(self):
        p=np.repeat(["2022","2023","2024"],30);z=np.zeros((90,10));cfg=config();cfg["inference"]["replicates"]=21;cfg["inference"]["blocks"]=[10]
        table=a.interval_rows(z,z,z,p,cfg)
        hedge=table[table.family=="hedge"]
        self.assertTrue((hedge.valid_replications==0).all())
        self.assertTrue(hedge.family_upper.isna().all())
        self.assertTrue((hedge.missing_reason!="").all())
        self.assertTrue((table[table.family=="full"].family_upper==0).all())
    def test_three_family_minimum_point_and_direction_evidence(self):
        cfg=config();periods=["2022","2023","2024"];point={};rows=[]
        for family,bases in [("full",a.BASELINES),("shape",a.BASELINES),("hedge",a.FINANCIAL)]:
            n=len(bases);value=-.06 if family=="hedge" else -.02
            point[family]={"macro":np.full(n,value),**{p:np.full(n,value if p!="2024" else .001) for p in periods}}
            for base in bases:rows.append(dict(family=family,baseline=base,statistic="macro",block=10,family_upper=-.00001))
        table=pd.DataFrame(rows);got=a.decision(table,point,periods,cfg)
        self.assertTrue(got["overall_proxy_advantage"])
        point["full"]["macro"][0]=-.005
        self.assertFalse(a.decision(table,point,periods,cfg)["full_matrix_signal"])
        point["full"]["macro"][0]=-.02;table.loc[table.family=="shape","family_upper"]=0.
        self.assertFalse(a.decision(table,point,periods,cfg)["overall_proxy_advantage"])
    def test_zero_target_and_fixed_tail_tie_all_rows_retained(self):
        f=fixture(zero=True);raw,e,ref,pred=a.ensemble(f,False)
        s,full,shape,risk,info=a.model_arrays(ref,pred)
        self.assertTrue(np.isfinite(full).all());np.testing.assert_array_equal(risk,np.zeros_like(risk))
        tail=a.tail_table(ref,full,shape,risk)
        self.assertEqual(len(tail),51);self.assertTrue(tail.trace_share.isna().all())
        self.assertTrue((tail.n_tail==1).all());self.assertEqual(len(ref),90)
        for period,group in ref.groupby("period"):
            self.assertTrue((tail[tail.period==period].tail_dates==group.date.iloc[0]).all())
    def test_config_cannot_silently_change_threshold_or_family(self):
        cfg=config();a.validate_config(cfg)
        for field,value in [("primary_minimum_point_delta",-.005),("replicates",100),("full_family_size",6)]:
            bad=copy.deepcopy(cfg);bad["inference"][field]=value
            with self.assertRaises(ValueError):a.validate_config(bad)
    def test_full_mock_pipeline_clock_and_artifacts(self):
        f=fixture(counts=(238,238,113));cfg=config()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/"outputs"/"run";out=root/"outputs"/"report";run.mkdir(parents=True)
            (root/"frozen_config.json").write_text(json.dumps(cfg),encoding="utf-8")
            (root/"FROZEN_PROTOCOL.md").write_text("synthetic protocol only",encoding="utf-8")
            (run/"manifest.json").write_text(json.dumps({"config_sha256":a.sha(root/"frozen_config.json")}),encoding="utf-8")
            (run/"status.json").write_text('{"status":"COMPLETE"}',encoding="utf-8")
            f.to_csv(run/"predictions_seeds.csv",index=False)
            with patch.object(a,"ROOT",root),patch.object(a,"ScientificClock",lambda task:ScientificClock(task,path=root/"clock.jsonl")):
                dec=a.run(run,out)
                self.assertFalse(dec["overall_proxy_advantage"])
                with self.assertRaises(FileExistsError):a.run(run,out)
            self.assertEqual(len(pd.read_csv(out/"paired_intervals.csv")),255)
            self.assertEqual(len(pd.read_csv(out/"tail_sensitivity.csv")),51)
            self.assertEqual(len(pd.read_csv(out/"model_metrics.csv")),20)
            self.assertEqual(len(pd.read_csv(out/"period_metrics.csv")),30)
            used,pending=spent(root/"clock.jsonl")
            self.assertGreater(used,0);self.assertFalse(pending)
            manifest=json.loads((out/"analysis_manifest.json").read_text())
            self.assertEqual(manifest["real_fits"],0)
            self.assertEqual(manifest["phase"],"exploration")

    def test_preflight_rejections_do_not_stop_market_or_start_clock(self):
        cfg=config()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run,out=mock_run_files(root,cfg)
            (run/"status.json").write_text('{"status":"RUNNING"}')
            with patch.object(a,"ROOT",root),patch.object(a,"ScientificClock",lambda task:ScientificClock(task,path=root/"clock.jsonl")):
                with self.assertRaises(ValueError):a.run(run,out)
                (run/"status.json").write_text('{"status":"COMPLETE"}')
                with self.assertRaises(ValueError):a.run(run,root/"not_canonical")
                out.mkdir()
                with self.assertRaises(FileExistsError):a.run(run,out)
                (root/"FROZEN_PROTOCOL.md").unlink()
                with self.assertRaises(ValueError):a.run(run,root/"outputs"/"new")
            self.assertFalse((root/"outputs"/"MARKET_STOP.json").exists())
            self.assertFalse((root/"clock.jsonl").exists())

    def test_scientific_failure_persists_global_stop_and_blocks_future_launch(self):
        cfg=config()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run,out=mock_run_files(root,cfg)
            with patch.object(a,"ROOT",root),patch.object(a,"ScientificClock",lambda task:ScientificClock(task,path=root/"clock.jsonl")):
                with patch.object(a,"analyze_frames",side_effect=ArithmeticError("synthetic numerical failure")):
                    with self.assertRaises(ArithmeticError):a.run(run,out)
                stop=root/"outputs"/"MARKET_STOP.json";original=stop.read_bytes()
                record=json.loads(original)
                self.assertEqual(record["error_type"],"ArithmeticError")
                self.assertEqual(record["stage"],"analysis")
                self.assertEqual(record["real_fits_in_analysis"],0)
                # This mirrors only the already-reviewed first stop-file check
                # in train.validate_launch; no train/optimizer is imported.
                def future_launch_guard():
                    if (root/"outputs"/"MARKET_STOP.json").exists():
                        raise RuntimeError("Prior scientific failure blocks later training")
                with self.assertRaises(RuntimeError):future_launch_guard()
                with patch.object(a,"analyze_frames",side_effect=ValueError("second failure")):
                    with self.assertRaises(ValueError):a.run(run,root/"outputs"/"report2")
                self.assertEqual(original,stop.read_bytes())
            _,pending=spent(root/"clock.jsonl");self.assertFalse(pending)
            self.assertFalse((root/"outputs"/"fit_budget.jsonl").exists())

    def test_csv_and_partial_output_failures_lock_canonical_stage(self):
        cfg=config()
        for failure in ("csv","write"):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);run,out=mock_run_files(root,cfg)
                with patch.object(a,"ROOT",root),patch.object(a,"ScientificClock",lambda task:ScientificClock(task,path=root/"clock.jsonl")):
                    if failure=="csv":
                        (run/"predictions_seeds.csv").write_text("bad_csv_only\n1\n")
                        with self.assertRaises(ValueError):a.run(run,out)
                    else:
                        fake=("exploration",{"partial":pd.DataFrame({"n":[1]})},{"fake":False})
                        def broken_write(frame,path,*args,**kwargs):
                            Path(path).write_text("PARTIAL")
                            raise OSError("synthetic disk write failure")
                        with patch.object(a,"analyze_frames",return_value=fake),patch.object(pd.DataFrame,"to_csv",new=broken_write):
                            with self.assertRaises(OSError):a.run(run,out)
                        self.assertEqual((out/"partial.csv").read_text(),"PARTIAL")
                self.assertTrue((root/"outputs"/"MARKET_STOP.json").exists())
                self.assertFalse((root/"outputs"/"fit_budget.jsonl").exists())

if __name__=="__main__":unittest.main()
