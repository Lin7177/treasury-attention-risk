import unittest
from unittest.mock import patch
import numpy as np
from numerical_v3 import qlike,derivatives,fit_har,predict_har,har_design,NumericalFailure,positive_guard
class NumericalTests(unittest.TestCase):
 def test_qlike_zero_and_mean_target(self):
  q=np.array([0.,2.,4.]);a=qlike(q,np.full(3,2.)).mean();self.assertTrue(np.isfinite(a));self.assertLess(a,qlike(q,np.ones(3)).mean());self.assertLess(a,qlike(q,np.full(3,4.)).mean())
 def test_analytic_gradient_hessian(self):
  rng=np.random.default_rng(1);z=np.c_[np.ones(20),rng.normal(size=(20,3))];q=rng.uniform(0,3,20);w=rng.normal(0,.1,4);f,g,h=derivatives(w,z,q);eps=1e-5;ng=[];nh=[]
  for j in range(4):
   e=np.eye(4)[j]*eps;ng.append((derivatives(w+e,z,q)[0]-derivatives(w-e,z,q)[0])/(2*eps));nh.append((derivatives(w+e,z,q)[1]-derivatives(w-e,z,q)[1])/(2*eps))
  np.testing.assert_allclose(g,ng,atol=1e-8);np.testing.assert_allclose(h,np.array(nh).T,atol=1e-8);self.assertGreaterEqual(np.linalg.eigvalsh(h).min(),-1e-12)
 def test_constant_and_zero_targets_kept(self):
  x=np.zeros((20,3));q=np.tile([0.,2.],10);s=fit_har(x,q);np.testing.assert_allclose(predict_har(s,x),1,atol=1e-9);self.assertLessEqual(s['trace'][-1]['gradient_inf'],1e-8)
 def test_feature_information_and_order(self):
  x=np.arange(2*20*5,dtype=float).reshape(2,20,5);own=har_design(x,True);multi=har_design(x,False);self.assertEqual(own.shape,(2,3));self.assertEqual(multi.shape,(2,15));np.testing.assert_allclose(own[:,0],np.log1p(x[:,-1,3]**2));np.testing.assert_allclose(own,multi[:,[3,8,13]])
 def test_unit_rescaling_preserves_features_and_loss_differences(self):
  rng=np.random.default_rng(4);x=rng.normal(size=(8,20,5));a=har_design(x,False,30.);b=har_design(x*10,False,3000.);np.testing.assert_allclose(a,b,atol=1e-14)
  q=rng.uniform(0,5,8);f=rng.uniform(.1,10,8);g=rng.uniform(.1,10,8);np.testing.assert_allclose(qlike(q,f)-qlike(q,g),qlike(q*100,f*100)-qlike(q*100,g*100),atol=1e-14)
 def test_fault_keeps_har_trace(self):
  x=np.arange(60,dtype=float).reshape(20,3);q=np.exp(np.linspace(-1,1,20))
  with patch('numerical_v3.np.linalg.solve',side_effect=np.linalg.LinAlgError('synthetic optimizer failure')):
   with self.assertRaises(NumericalFailure) as caught:fit_har(x,q)
  self.assertGreaterEqual(len(caught.exception.trace),1);self.assertEqual(caught.exception.diagnostics['role'],'har_optimizer')
 def test_floor_fault_records_role_counts(self):
  with self.assertRaises(NumericalFailure) as caught:positive_guard(np.array([1.,1e-8,0.,np.nan]),'validation')
  self.assertEqual(caught.exception.diagnostics['raw_floor_hits'],2);self.assertEqual(caught.exception.diagnostics['nonfinite_raw'],1);self.assertEqual(caught.exception.diagnostics['role'],'validation')
 def test_invalid_forecast_rejected(self):
  with self.assertRaises(ValueError):qlike(np.array([1.]),np.array([0.]))
  with self.assertRaises(ValueError):fit_har(np.zeros((3,1)),np.zeros(3))
if __name__=='__main__':unittest.main()
