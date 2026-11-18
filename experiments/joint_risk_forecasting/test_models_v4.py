"""Synthetic V4 scorer guards only; market data never accessed."""
import copy
import math
import unittest
import torch
from models_v4 import (build_model, MODEL_NAMES, key_query_features, fixed_matrix,
                       covariance_from_weights, kernel_bandwidth_statistics,
                       time_log_prior, FEATURE_EPSILON)

torch.set_num_threads(1)


def synthetic_x(n=12):
    return torch.randn(n, 20, 5, generator=torch.Generator().manual_seed(76201), dtype=torch.float64)


def loss(target, forecast):
    chol = torch.linalg.cholesky(forecast)
    logdet = 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    solved = torch.cholesky_solve(target, chol)
    return (logdet + torch.diagonal(solved, dim1=-2, dim2=-1).sum(-1)).mean()


class MatrixGuards(unittest.TestCase):
    def setUp(self):
        self.x = synthetic_x()
        self.c0 = torch.tensor([[1,.3,.2],[.3,1,.4],[.2,.4,1]], dtype=torch.float64)

    def test_parameter_counts_initial_function_and_dtype(self):
        models = [build_model(n, 2026110541) for n in MODEL_NAMES]
        self.assertEqual([m.parameter_summary()["total"] for m in models], [19,283,299])
        expected, ew = fixed_matrix(self.x, self.c0, "EWMA", return_weights=True)
        for m in models:
            h, w = m(self.x, self.c0, return_weights=True)
            self.assertTrue(torch.equal(h, expected))
            self.assertTrue(torch.equal(w, ew))
            self.assertTrue(all(p.dtype == torch.float64 for p in m.parameters()))
            self.assertEqual(m.parameter_summary()["structurally_inactive"], 0)
        self.assertEqual(models[1].key_projection.bias, None)
        self.assertEqual(models[2].output.bias, None)

    def test_rng_streams_and_replay(self):
        torch.manual_seed(77)
        before = torch.get_rng_state().clone()
        for name in ("MLP","Content","Static","Content"):
            m = build_model(name, 2026110541)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            other = build_model(name, 2026110541)
            self.assertEqual(m.initialization_sha256(), other.initialization_sha256())
            other.load_state_dict(copy.deepcopy(m.state_dict()))
            self.assertTrue(torch.equal(m(self.x,self.c0), other(self.x,self.c0)))
        self.assertNotEqual(build_model("Content",1).initialization_sha256(),
                            build_model("Content",2).initialization_sha256())

    def test_exact_features_latest_five_and_global_sign_invariance(self):
        k,q=key_query_features(self.x)
        self.assertEqual(tuple(k.shape), (12,20,16))
        self.assertTrue(torch.equal(q,k[:,-5:].mean(1)))
        energy=self.x.square().sum(-1)
        self.assertTrue(torch.equal(k[:,:,0], self.x[:,:,0].square()/(energy+FEATURE_EPSILON)))
        self.assertTrue(torch.equal(k[:,:,1], self.x[:,:,0]*self.x[:,:,1]/(energy+FEATURE_EPSILON)))
        self.assertTrue(torch.equal(k[:,:,-1],torch.log1p(energy)))
        changed=self.x.clone()
        changed[:,0]*=3
        kc,qc=key_query_features(changed)
        self.assertTrue(torch.equal(q,qc))
        self.assertTrue(torch.equal(k[:,1:],kc[:,1:]))
        kn,qn=key_query_features(-self.x)
        self.assertTrue(torch.equal(k,kn))
        self.assertTrue(torch.equal(q,qn))
        for name in MODEL_NAMES:
            m=build_model(name,33)
            self.assertTrue(torch.equal(m(self.x,self.c0),m(-self.x,self.c0)))

    def test_value_target_indices_factor_five_and_zero(self):
        w=torch.zeros(12,20,dtype=torch.float64)
        w[:,3]=1
        got=covariance_from_weights(self.x,self.c0,w)
        v=self.x[:,3,2:5]
        expected=5*(.9*v[:,:,None]*v[:,None,:]+.1*self.c0)
        self.assertTrue(torch.allclose(got,expected,atol=1e-14,rtol=1e-14))
        zero=torch.zeros(1,20,5,dtype=torch.float64)
        for name in MODEL_NAMES:
            h=build_model(name,1)(zero,self.c0)
            self.assertTrue(torch.allclose(h,.5*self.c0.unsqueeze(0),atol=1e-15,rtol=0))
            self.assertEqual(int(torch.linalg.cholesky_ex(h).info),0)
            self.assertTrue(torch.isfinite(loss(torch.zeros_like(h),h)))

    def test_content_and_mlp_are_live_after_first_steps(self):
        v=self.x[:,0,2:5]
        target=5*v[:,:,None]*v[:,None,:]+.2*torch.eye(3,dtype=torch.float64)
        for name in MODEL_NAMES:
            model=build_model(name,2026110541)
            opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
            for step in range(4):
                opt.zero_grad()
                value=loss(target,model(self.x,self.c0))
                value.backward()
                if step==0 and name=="Content":
                    self.assertEqual(torch.count_nonzero(model.key_projection.weight.grad).item(),0)
                    self.assertGreater(torch.count_nonzero(model.query_projection.weight.grad).item(),0)
                if step==0 and name=="MLP":
                    self.assertEqual(torch.count_nonzero(model.hidden.weight.grad).item(),0)
                    self.assertGreater(torch.count_nonzero(model.output.weight.grad).item(),0)
                if step>=1:
                    for key,p in model.named_parameters():
                        self.assertIsNotNone(p.grad,key)
                        self.assertTrue(torch.isfinite(p.grad).all(),key)
                        self.assertGreater(torch.count_nonzero(p.grad).item(),0,key)
                opt.step()
            h,w=model(self.x,self.c0,return_weights=True)
            self.assertTrue(torch.isfinite(h).all())
            self.assertTrue(torch.allclose(w.sum(-1),torch.ones(12,dtype=torch.float64),atol=1e-14))

    def test_counterfactual_static_and_uniform_same_values_prior(self):
        model=build_model("Content",1)
        with torch.no_grad():
            model.position_delta.copy_(torch.linspace(-.2,.2,19,dtype=torch.float64))
            model.query_projection.weight.fill_(.1)
        full,w=model(self.x,self.c0,return_weights=True)
        static,sw=model(self.x,self.c0,force_static=True,return_weights=True)
        other=build_model("Static",9)
        other.position_delta.data.copy_(model.position_delta.data)
        self.assertTrue(torch.equal(static,other(self.x,self.c0)))
        uniform=model(self.x,self.c0,force_uniform=True)
        self.assertTrue(torch.equal(uniform,fixed_matrix(self.x,self.c0,"Uniform")))
        self.assertFalse(torch.allclose(full,static))
        with self.assertRaises(ValueError):
            model(self.x,self.c0,force_uniform=True,force_static=True)

    def test_kernel_one_training_geometry_median_and_inputs(self):
        stats=kernel_bandwidth_statistics(self.x)
        k,q=key_query_features(self.x)
        distances=(k-q[:,None,:]).square().sum(-1)
        expected=float(torch.quantile(distances[distances>0],.5))
        self.assertEqual(stats["bandwidth_squared"],expected)
        self.assertEqual(stats["query_key_pairs"],12*20)
        self.assertEqual(stats["optimizer_fits"],0)
        h,w=fixed_matrix(self.x,self.c0,"Kernel",expected,True)
        logits=time_log_prior()[None,:]-distances/(2*expected)
        self.assertTrue(torch.equal(w,torch.softmax(logits,-1)))
        self.assertTrue(torch.isfinite(h).all())
        self.assertEqual(kernel_bandwidth_statistics(torch.zeros_like(self.x))["bandwidth_squared"],1)
        for bad in (None,0,-1,float("nan")):
            with self.assertRaises(ValueError):
                fixed_matrix(self.x,self.c0,"Kernel",bad)

    def test_coordinate_units_qlike_and_common_diagonal(self):
        h=build_model("Content",2)(self.x,self.c0)
        v=self.x[:,4,2:5]
        target=5*v[:,:,None]*v[:,None,:]
        s=torch.tensor([2,3,7],dtype=torch.float64)
        hh=h*s[None,:,None]*s[None,None,:]
        tt=target*s[None,:,None]*s[None,None,:]
        self.assertAlmostEqual(float((loss(tt,hh)-loss(target,h)).detach()),float(2*torch.log(s).sum()),places=11)
        other=fixed_matrix(self.x,self.c0,"Uniform")
        self.assertAlmostEqual(float((loss(tt,hh)-loss(tt,other*s[None,:,None]*s[None,None,:])).detach()),
                               float((loss(target,h)-loss(target,other)).detach()),places=11)
        diagonal=torch.diagonal(h,dim1=-2,dim2=-1).sqrt()
        r=h/diagonal[:,:,None]/diagonal[:,None,:]
        d=torch.tensor([2.,3.,4.],dtype=torch.float64)
        reconstructed=r*d[None,:,None]*d[None,None,:]
        self.assertTrue(torch.allclose(torch.diagonal(reconstructed,dim1=-2,dim2=-1),
                                       d.square().expand(12,-1),atol=1e-12))

    def test_independent_day_sign_symmetry_and_all_overflow_paths(self):
        changed=self.x.clone()
        changed[:,[0,3,17]] *= -1
        for name in MODEL_NAMES:
            m=build_model(name,1)
            self.assertTrue(torch.equal(m(self.x,self.c0),m(changed,self.c0)))
        bandwidth=kernel_bandwidth_statistics(self.x)["bandwidth_squared"]
        for name in ("Uniform","EWMA","Kernel"):
            self.assertTrue(torch.equal(fixed_matrix(self.x,self.c0,name,bandwidth),
                                        fixed_matrix(changed,self.c0,name,bandwidth)))
        overflowing=torch.full((1,20,5),1e200,dtype=torch.float64)
        w=torch.full((1,20),.05,dtype=torch.float64)
        with self.assertRaises(FloatingPointError):
            covariance_from_weights(overflowing,self.c0,w)
        with self.assertRaises(FloatingPointError):
            fixed_matrix(overflowing,self.c0,"EWMA")
        with self.assertRaises(FloatingPointError):
            build_model("Content",1)(overflowing,self.c0,force_uniform=True)

    def test_invalid_shapes_prior_and_extreme_numerics(self):
        m=build_model("Content",1)
        for x in (torch.zeros(20,5,dtype=torch.float64),torch.zeros(0,20,5,dtype=torch.float64),
                  self.x.float(),torch.full((1,20,5),float("nan"),dtype=torch.float64)):
            with self.assertRaises(ValueError):m(x,self.c0)
        for c in (torch.zeros(3,3,dtype=torch.float64),torch.ones(2,2,dtype=torch.float64),
                  torch.tensor([[1.,2.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=torch.float64)):
            with self.assertRaises(ValueError):m(self.x,c)
        extreme=self.x*1e100
        h=m(extreme,self.c0)
        self.assertTrue(torch.isfinite(h).all())
        overflowing=self.x*1e200
        with self.assertRaises(ValueError):m(overflowing,self.c0)


if __name__=="__main__":
    unittest.main()
