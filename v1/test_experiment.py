import unittest
import numpy as np
import pandas as pd
from experiment import metrics,logistic,sigmoid,feature
class DataTests(unittest.TestCase):
 def test_flat_is_wrong_but_not_in_brier_ba(self):
  d=pd.DataFrame({'direction':[1,-1,0]})
  m=metrics([.8,.2,.2],d)
  self.assertEqual(m['accuracy'],2/3);self.assertEqual(m['balanced_accuracy'],1);self.assertAlmostEqual(m['brier'],.04)
 def test_logistic_stationarity_and_constant_feature(self):
  x=np.ones((10,2));y=np.array([1.]*7+[0.]*3);mu,sd,w=logistic(x,y)
  np.testing.assert_allclose(sigmoid(np.c_[np.ones(10),(x-mu)/sd]@w),.7,atol=1e-7)
 def test_own_features_have_no_other_tenor(self):
  x=np.ones((2,20,5));s=np.zeros((2,3));a=feature(x,s,True);x[:,:,0]=100
  np.testing.assert_equal(a,feature(x,s,True));self.assertEqual(a.shape,(2,4))
