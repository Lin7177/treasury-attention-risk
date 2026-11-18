import unittest
import numpy as np
import pandas as pd
from numerical import metrics,macro_metrics,fit_logit,predict_logit
class MetricTests(unittest.TestCase):
 def test_flat(self):
  d=pd.DataFrame({'direction':[1,-1,0]});m=metrics([.8,.2,.2],d)
  self.assertEqual(m['accuracy'],2/3);self.assertEqual(m['balanced_accuracy'],1);self.assertAlmostEqual(m['brier'],.04)
 def test_missing_class(self):
  d=pd.DataFrame({'direction':[1,1], 'date':['2022-01-01','2022-01-02']});self.assertIsNone(macro_metrics([.8,.8],d))
 def test_prior_pooling(self):
  d=pd.DataFrame({'direction':[1]*8+[-1]*2+[1]*2+[-1]*8,'date':['2022-01-01']*10+['2023-01-01']*10});p=np.array([.6]*10+[.4]*10)
  self.assertEqual(metrics(p,d)['balanced_accuracy'],.8);self.assertEqual(macro_metrics(p,d),.5)
 def test_constant_feature_logit(self):
  x=np.ones((10,3));m=fit_logit(x,np.array([1.]*7+[0.]*3));np.testing.assert_allclose(predict_logit(m,x),.7,atol=1e-7)
