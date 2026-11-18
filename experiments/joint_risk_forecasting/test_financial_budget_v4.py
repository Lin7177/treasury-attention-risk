import unittest,tempfile,json
from pathlib import Path
import numpy as np
import financial_v4 as f
import budget_v4 as b

def finite_difference(function,x,epsilon=1e-6):
    out=[]
    for k in range(len(x)):
        delta=np.zeros(len(x));delta[k]=epsilon
        out.append((function(x+delta)[0]-function(x-delta)[0])/(2*epsilon))
    return np.array(out)
class FinancialTests(unittest.TestCase):
    def test_garch_gradient_and_missing(self):
        rng=np.random.default_rng(23);r=rng.normal(size=91);r[[3,40]]=np.nan;x=np.array([-2.,-.2,2.])
        np.testing.assert_allclose(f.garch_objective(x,r)[1],finite_difference(lambda u:f.garch_objective(u,r),x),rtol=2e-5,atol=1e-7)
    def test_dcc_gradient_and_partial_missing(self):
        rng=np.random.default_rng(11);z=rng.normal(size=(75,3));z[20,0]=np.nan
        qbar=.99*(z[np.isfinite(z).all(axis=1)].T@z[np.isfinite(z).all(axis=1)])/74+.01*np.eye(3)
        x=np.array([-.3,2.1])
        np.testing.assert_allclose(f.dcc_objective(x,z,qbar)[1],finite_difference(lambda u:f.dcc_objective(u,z,qbar),x),rtol=2e-5,atol=1e-7)
    def test_garch_causal(self):
        r=np.arange(50)/50;changed=r.copy();changed[30:]+=99
        np.testing.assert_array_equal(f.garch_filter(r,.1,.05,.85)[:31],f.garch_filter(changed,.1,.05,.85)[:31])
    def test_dcc_causal(self):
        z=np.random.default_rng(2).normal(size=(50,3));other=z.copy();other[30:]*=90
        one=f.dcc_filter(z,np.eye(3),.04,.9);two=f.dcc_filter(other,np.eye(3),.04,.9)
        np.testing.assert_array_equal(one['q'][:31],two['q'][:31])
    def test_missing_garch_expected_step(self):
        h=f.garch_filter(np.r_[np.nan,np.ones(31)],.2,.1,.6)
        self.assertAlmostEqual(h[1],.9)
    def test_matrix_loss_unit_difference(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(10,3,3));a=x@x.transpose(0,2,1);one=a+.3*np.eye(3);two=a+.6*np.eye(3)
        d=np.diag([.1,100,2])
        np.testing.assert_allclose(f.matrix_qlike(d@a@d,d@one@d)-f.matrix_qlike(d@a@d,d@two@d),f.matrix_qlike(a,one)-f.matrix_qlike(a,two),atol=2e-13)
    def test_qmc_static_exact(self):
        q=np.array([[[1,.5,.2],[.5,1,.3],[.2,.3,1]]]);g=[dict(omega=1.,alpha=0.,beta=0.) for _ in range(3)]
        out=f.forecast_dcc_five(np.ones((1,3)),q,g,dict(parameters=dict(alpha=0.,beta=0.),qbar=q[0].tolist()),exponent=6)
        np.testing.assert_allclose(out,5*q,atol=1e-12)
    def test_convex_har_synthetic(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(120,20));target=np.exp(.1*np.mean(x*x,axis=1))
        result=f.fit_har(x,target)
        self.assertLessEqual(result['gradient_inf'],1e-8)
        self.assertTrue((f.predict_har(x,result)>0).all())
    def test_garch_synthetic_fit(self):
        r=np.random.default_rng(1).normal(size=180);fit=f.fit_garch(r)
        self.assertLessEqual(fit['projected_gradient'],1e-4)
        self.assertLess(fit['parameters']['alpha']+fit['parameters']['beta'],1)
    def test_matrix_contract_rejects_invalid(self):
        identity=np.eye(3)[None]
        malformed=identity.copy();malformed[0,0,1]=100
        for a,h in [(identity,malformed),(-identity,identity),(identity*np.nan,identity),(identity,-identity)]:
            with self.subTest(a=a,h=h),self.assertRaises((ValueError,np.linalg.LinAlgError)):
                f.matrix_qlike(a,h)
    def test_optimizer_deadline_synthetic(self):
        with self.assertRaises(f.FitError):
            f.fit_garch(np.ones(100),deadline=0.)
    def test_qmc_request_memory_cap(self):
        with self.assertRaises(ValueError):
            f.controlled_dcc_five(np.ones((9,3)),np.broadcast_to(np.eye(3),(9,3,3)),[],{})
class BudgetTests(unittest.TestCase):
    def test_failure_consumed_and_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'journal.jsonl';b.claim('one','exploration','garch','synthetic',p);b.finish('one','failed',1,path=p)
            with self.assertRaises(RuntimeError):b.claim('two','exploration','garch','synthetic',p)
            self.assertEqual(b.summary(p)['claims'],1)
    def test_cross_run_same_job_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'journal.jsonl';b.claim('one','exploration','garch','synthetic',p);b.finish('one','complete',1,path=p)
            with self.assertRaises(RuntimeError):b.claim('one','exploration','garch','different-run',p)
    def test_phase_quota(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'journal.jsonl'
            for i in range(3):b.claim(str(i),'exploration','dcc','synthetic',p);b.finish(str(i),'complete',1,path=p)
            with self.assertRaises(RuntimeError):b.claim('excess','exploration','dcc','synthetic',p)
    def test_invalid_duration_is_rejected_without_finish_event(self):
        for seconds in [float("nan"),float("inf"),-1,True]:
            with self.subTest(seconds=seconds),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/"journal.jsonl";b.claim("one","exploration","garch","synthetic",p)
                with self.assertRaises(ValueError):b.finish("one","complete",seconds,path=p)
                self.assertEqual(len(b.load(p)),1)
    def test_nonfinite_failure_diagnostic_does_not_erase_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"journal.jsonl";b.claim("one","exploration","garch","synthetic",p)
            b.finish("one","failed",.1,dict(pg=float("nan")),p)
            self.assertEqual(b.summary(p)["failed"],1)
            self.assertIsNone(b.load(p)[-1]["details"]["original_diagnostics"]["pg"])
    def test_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'journal.jsonl';b.claim('one','exploration','garch','synthetic',p)
            row=json.loads(p.read_text());row['job']='tamper';p.write_text(json.dumps(row)+'\n')
            with self.assertRaises(RuntimeError):b.load(p)
if __name__=='__main__':unittest.main()
