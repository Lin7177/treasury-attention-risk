"""Synthetic contract guards; no market optimization or decomposition."""
import unittest
import numpy as np
import pandas as pd
from data_v3 import build_dataset, split_data, fit_input_transform, transform_x

class DataContractTests(unittest.TestCase):
    def fixture(self, n=1600):
        dates = pd.bdate_range("2018-01-01", periods=n)
        levels = 2 + np.cumsum(np.random.default_rng(11).normal(0,.01,(n,5)),axis=0)
        return dates, levels

    def test_original_axis_endpoint_and_window(self):
        dates,y=self.fixture(100)
        d,x,delta=build_dataset(dates,y)
        r=d.iloc[0]; t=int(r["index"])
        self.assertEqual(t,21)
        self.assertEqual(r["label_end"],dates[t+4])
        self.assertEqual(r["information_end"],dates[t-1])
        np.testing.assert_array_equal(x[0],delta[t-20:t])
        self.assertAlmostEqual(r["q5"],np.square(delta[t:t+5,3]).sum())
        self.assertEqual(d.attrs["flow"]["eligible"],len(d))

    def test_future_middle_target_missing_but_external_unneeded(self):
        dates,y=self.fixture(100)
        bad=y.copy(); bad[52,3]=np.nan
        d,_,_=build_dataset(dates,bad)
        self.assertNotIn(50,d["index"].tolist())
        external=y.copy(); external[52,0]=np.nan
        d,_,_=build_dataset(dates,external)
        self.assertIn(50,d["index"].tolist())

    def test_history_missing_not_compressed(self):
        dates,y=self.fixture(100)
        y[40,0]=np.nan
        d,_,_=build_dataset(dates,y)
        self.assertNotIn(50,d["index"].tolist())
        self.assertIn(65,d["index"].tolist())

    def test_zero_target_and_constant_scale(self):
        dates,_=self.fixture(100)
        d,x,delta=build_dataset(dates,np.ones((100,5)))
        self.assertTrue((d["q5"]==0).all())
        state=fit_input_transform(delta,d,np.arange(5))
        self.assertEqual(state["zero_std_channels"],list(range(10)))
        np.testing.assert_array_equal(transform_x(x,state),0)

    def test_deduplicated_training_days(self):
        dates,y=self.fixture(100)
        d,x,delta=build_dataset(dates,y)
        state=fit_input_transform(delta,d,np.arange(3))
        unique=np.arange(1,23)
        self.assertEqual(state["unique_day_indices"],unique.tolist())
        z=np.concatenate([delta[unique],delta[unique]**2],axis=1)
        np.testing.assert_allclose(state["mean10"],z.mean(axis=0))
        np.testing.assert_allclose(state["scale10"],z.std(axis=0))
        self.assertEqual(transform_x(x,state).shape,(len(d),20,10))

    def test_future_mutation_does_not_change_train_statistics(self):
        dates,y=self.fixture()
        d,_,delta=build_dataset(dates,y)
        tr,va,ev=split_data(d,2022)
        state=fit_input_transform(delta,d,tr)
        altered=y.copy()
        altered[np.asarray(dates.year)>=2022]+=100
        d2,_,delta2=build_dataset(dates,altered)
        tr2,va2,ev2=split_data(d2,2022)
        np.testing.assert_array_equal(tr,tr2)
        np.testing.assert_array_equal(va,va2)
        state2=fit_input_transform(delta2,d2,tr2)
        self.assertEqual(state,state2)
        np.testing.assert_array_equal(d.iloc[tr]["q5"],d2.iloc[tr2]["q5"])
        self.assertLess(d.iloc[tr]["label_end"].max(),d.iloc[va[0]]["date"])
        self.assertLess(d.iloc[va]["label_end"].max(),d.iloc[ev[0]]["date"])

    def test_date_contract(self):
        dates,y=self.fixture(100)
        bad=list(dates); bad[2]=bad[1]
        with self.assertRaises(ValueError): build_dataset(bad,y)
        with self.assertRaises(ValueError):
            build_dataset(pd.bdate_range("2025-01-01",periods=100),y)


    def test_prior_strict_maturity_and_future_label_invariance(self):
        from prior_diagnostic import prior_states
        ledger=pd.DataFrame(dict(index=[1,2,3,4],
            date=pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03","2020-01-04"]),
            label_end=pd.to_datetime(["2020-01-06","2020-01-07","2020-01-08","2020-01-09"]),
            direction=[1,-1,1,-1]))
        requests=pd.DataFrame(dict(index=[10],date=pd.to_datetime(["2020-01-08"])))
        state=prior_states(ledger,requests,.4)
        self.assertEqual(state.iloc[0]["history_count"],2)
        self.assertAlmostEqual(state.iloc[0]["probability_up"],9/22)
        changed=ledger.copy();changed.loc[changed["label_end"]>=pd.Timestamp("2020-01-08"),"direction"]=0
        pd.testing.assert_frame_equal(state,prior_states(changed,requests,.4))

    def test_prior_reconstruction_not_same_as_state_change(self):
        from prior_diagnostic import prior_states
        ledger=pd.DataFrame(dict(index=[1],date=pd.to_datetime(["2020-01-01"]),
            label_end=pd.to_datetime(["2020-01-06"]),direction=[1]))
        req=pd.DataFrame(dict(index=[2,3],date=pd.to_datetime(["2020-01-08","2020-01-09"])))
        states=prior_states(ledger,req,.4)
        self.assertEqual(len(states),2)
        self.assertEqual(states.iloc[1]["new_matured_since_previous"],0)
        self.assertFalse(states.iloc[1]["q_changed_since_previous"])

    def test_prior_flat_denominators(self):
        from prior_diagnostic import metrics
        result=metrics([1,0,-1],[.8,.2,.1])
        self.assertEqual(result["n_all"],3)
        self.assertEqual(result["n_nonflat"],2)
        self.assertAlmostEqual(result["accuracy_all"],2/3)
        self.assertAlmostEqual(result["brier_nonflat"],.025)
        self.assertEqual(result["balanced_accuracy_nonflat"],1)


    def test_prior_exact_legacy_long_model_identifiers(self):
        from prior_diagnostic import select_saved_models
        saved=pd.DataFrame(dict(model=["N0_OwnMaskedFixed","N3_ContentCross","N0_Fake","N3","N1_Fixed"],
                                probability_up=[.1,.2,.3,.4,.5]))
        selected=select_saved_models(saved)
        self.assertEqual(selected["model"].tolist(),["N0","N3"])
        self.assertEqual(selected["probability_up"].tolist(),[.1,.2])

if __name__=="__main__":
    unittest.main()
