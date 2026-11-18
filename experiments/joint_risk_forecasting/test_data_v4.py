"""Synthetic V4 data guards, without market optimization or network."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import data_v4 as data

class DataV4Tests(unittest.TestCase):
    def fixture(self,n=1600):
        dates=pd.bdate_range("2018-01-01",periods=n)
        levels=2+np.cumsum(np.random.default_rng(41).normal(.001,.01,(n,5)),axis=0)
        return dates,levels

    def test_original_axis_window_and_three_target_matrix(self):
        dates,y=self.fixture(100)
        d,x,a,axis,delta=data.build_dataset(dates,y)
        t=int(d.iloc[0]["index"])
        self.assertEqual(t,21)
        self.assertEqual(d.iloc[0]["information_end"],dates[t-1])
        self.assertEqual(d.iloc[0]["label_end"],dates[t+4])
        np.testing.assert_array_equal(x[0],delta[t-20:t])
        future=delta[t:t+5][:,[2,3,4]]
        np.testing.assert_allclose(a[0],future.T@future)
        np.testing.assert_allclose(a,a.swapaxes(1,2))
        self.assertEqual(a.shape,(len(d),3,3))
        self.assertEqual(d.attrs["flow"]["eligible"],len(d))

    def test_future_1y_3y_missing_remains_eligible(self):
        dates,y=self.fixture(100)
        y[52,0]=np.nan;y[53,1]=np.nan
        d,_,a,_,_=data.build_dataset(dates,y)
        self.assertIn(50,d["index"].tolist())
        self.assertTrue(np.isfinite(a).all())

    def test_future_target_middle_missing_rejected(self):
        dates,y=self.fixture(100)
        for col in (2,3,4):
            bad=y.copy();bad[52,col]=np.nan
            d,_,_,_,_=data.build_dataset(dates,bad)
            self.assertNotIn(50,d["index"].tolist())

    def test_historical_any_input_missing_not_compressed(self):
        dates,y=self.fixture(100)
        y[40,0]=np.nan
        d,_,_,_,_=data.build_dataset(dates,y)
        self.assertNotIn(50,d["index"].tolist())
        self.assertIn(65,d["index"].tolist())
        self.assertEqual(d.loc[d["index"]==65,"date"].iloc[0],dates[65])

    def test_zero_target_is_not_dropped(self):
        dates,y=self.fixture(100)
        y[49:55,2:5]=y[49,2:5]
        d,_,a,_,_=data.build_dataset(dates,y)
        pos=d.index[d["index"]==50][0]
        np.testing.assert_array_equal(a[pos],np.zeros((3,3)))

    def test_training_rms_deduplicated_and_not_centered(self):
        dates,y=self.fixture(100)
        d,x,a,_,delta=data.build_dataset(dates,y)
        st=data.fit_training_statistics(delta,d,np.arange(3))
        ids=np.arange(1,23)
        self.assertEqual(st["unique_day_indices"],ids.tolist())
        rms=np.sqrt((delta[ids]**2).mean(axis=0))
        np.testing.assert_allclose(st["rms5"],rms)
        self.assertFalse(st["centered"])
        self.assertGreater(np.abs((delta[ids]/rms).mean(axis=0)).max(),.01)
        z=data.normalize_inputs(x,st)
        np.testing.assert_allclose(z,x/rms)
        self.assertEqual(z.shape,(len(d),20,5))
        targetrms=rms[[2,3,4]]
        np.testing.assert_allclose(data.normalize_targets(a,st),a/np.outer(targetrms,targetrms))

    def test_daily_prior_and_fixed_shrinkage(self):
        dates,y=self.fixture(100)
        d,_,_,_,delta=data.build_dataset(dates,y)
        st=data.fit_training_statistics(delta,d,np.arange(8))
        ids=st["unique_day_indices"]
        z=delta[ids]/np.asarray(st["rms5"])
        v=z[:,[2,3,4]]
        expected=v.T@v/len(v)
        np.testing.assert_allclose(st["Ctrain"],expected)
        np.testing.assert_allclose(st["C0"],.99*expected+.01*np.eye(3))
        np.testing.assert_allclose(np.diag(st["C0"]),np.ones(3))
        self.assertEqual(st["prior_units"],"one_day_normalized_outer_product")
        self.assertGreaterEqual(np.linalg.eigvalsh(np.asarray(st["C0"])).min(),.01-1e-12)

    def test_zero_rms_is_explicit_failure(self):
        dates,_=self.fixture(100)
        d,_,_,_,delta=data.build_dataset(dates,np.ones((100,5)))
        with self.assertRaisesRegex(ValueError,"RMS"):
            data.fit_training_statistics(delta,d,np.arange(3))

    def test_future_mutation_does_not_change_train_statistics(self):
        dates,y=self.fixture()
        d,_,a,_,delta=data.build_dataset(dates,y)
        tr,va,ev=data.split_data(d,2022)
        st=data.fit_training_statistics(delta,d,tr)
        changed=y.copy();changed[dates.year>=2022]+=100
        d2,_,a2,_,delta2=data.build_dataset(dates,changed)
        tr2,va2,ev2=data.split_data(d2,2022)
        np.testing.assert_array_equal(tr,tr2)
        np.testing.assert_array_equal(va,va2)
        self.assertEqual(st,data.fit_training_statistics(delta2,d2,tr2))
        np.testing.assert_array_equal(a[tr],a2[tr2])
        self.assertLess(d.iloc[tr].label_end.max(),d.iloc[va[0]].date)
        self.assertLess(d.iloc[va].label_end.max(),d.iloc[ev[0]].date)

    def test_traditional_training_axis_includes_labels_not_compressed(self):
        dates,y=self.fixture()
        d,_,_,axis,delta=data.build_dataset(dates,y)
        tr,_,_=data.split_data(d,2022)
        ids=data.training_axis_indices(d,tr,len(axis))
        self.assertEqual(ids[0],int(d.iloc[tr]["index"].min())-20)
        self.assertEqual(ids[-1],int(d.iloc[tr]["index"].max())+4)
        np.testing.assert_array_equal(np.diff(ids),np.ones(len(ids)-1,dtype=int))

    def test_invalid_dates_fail(self):
        dates,y=self.fixture(100)
        changed=list(dates);changed[2]=changed[1]
        with self.assertRaises(ValueError):data.build_dataset(changed,y)
        with self.assertRaises(ValueError):
            data.build_dataset(pd.bdate_range("2025-01-01",periods=100),y)

    def test_synthetic_us_parser_filters_before_numeric(self):
        frame=pd.DataFrame({"Date":["06/27/2024","06/28/2024","07/01/2024"],
            **{c:["1.0","2.0","UNAUTHORIZED_NOT_A_NUMBER"] for c in ("1 Yr","3 Yr","5 Yr","7 Yr","10 Yr")}})
        original=pd.to_numeric
        seen=[]
        def guarded(series,*args,**kwargs):
            self.assertNotIn("UNAUTHORIZED_NOT_A_NUMBER",series.tolist())
            seen.append(len(series))
            return original(series,*args,**kwargs)
        with patch.object(data.pd,"to_numeric",new=guarded):
            dates,levels,metadata=data.parse_us_text_frame(frame,"2024-06-30")
        self.assertEqual(seen,[2]*5)
        self.assertEqual(levels.shape,(2,5))
        self.assertEqual(dates[-1],pd.Timestamp("2024-06-28"))
        self.assertTrue(metadata["cutoff_before_numeric_conversion"])
        self.assertIn("synthetic",metadata["template_scope"])

if __name__=="__main__":
    unittest.main()
