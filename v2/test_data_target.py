"""Synthetic-only contract tests. No market PCA fit or real classifier fit."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import data_target as data


class LoaderTests(unittest.TestCase):
    def test_sealed_values_not_numerically_parsed(self):
        dates=pd.date_range('2024-05-01',periods=50).strftime('%Y/%m/%d').tolist()+['2024/07/01']
        frame=pd.DataFrame({'date':dates,**{c:['3.0']*50+['SEALED_SENTINEL'] for c in data.COLS}})
        original=pd.to_numeric;calls=[]
        def guard(values,*args,**kwargs):
            self.assertNotIn('SEALED_SENTINEL',set(values));calls.append(len(values))
            return original(values,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.csv';frame.to_csv(p,index=False)
            with patch.object(data.pd,'to_numeric',new=guard):
                d,x,sp,raw=data.load_data(p,strict=False)
        self.assertEqual(calls,[50]*5)
        self.assertEqual(raw.shape,(50,5));self.assertEqual(x.shape,(24,20,5));self.assertEqual(sp.shape,(24,4))
        self.assertEqual(d.date.iloc[0],'2024-05-22');self.assertEqual(d.label_end.iloc[0],'2024-05-27')

    def test_lag_endpoint_spread_and_no_axis_reordering(self):
        dates=pd.date_range('2020-01-01',periods=40).strftime('%Y-%m-%d')
        levels=np.arange(40)[:,None]*np.array([1,2,3,4,5])[None,:]/100
        d,x,sp,raw=data.build_samples(dates,levels)
        np.testing.assert_allclose(x[0],np.tile([1,2,3,4,5],(20,1)),atol=1e-12)
        np.testing.assert_allclose(sp[0],[60,40,20,-20],atol=1e-12)
        self.assertEqual(d.iloc[0]['index'],21);self.assertAlmostEqual(d.iloc[0].change_bp,20.,places=12)
        changed=levels.copy();changed[21,3]+=10
        d2,x2,sp2,_=data.build_samples(dates,changed)
        np.testing.assert_equal(x[0],x2[0]);np.testing.assert_equal(sp[0],sp2[0])
        self.assertNotEqual(d.iloc[0].direction,d2.iloc[0].direction)
        missing=levels.copy();missing[10,2]=np.nan
        dm,_,_,_=data.build_samples(dates,missing)
        self.assertEqual(dm.iloc[0]['index'],32)
        self.assertEqual(dm.iloc[0].label_end,dates[37])

    def test_duplicate_date_rejected(self):
        frame=pd.DataFrame({'date':['2020/01/01','2020/01/01'],**{c:[1,2] for c in data.COLS}})
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.csv';frame.to_csv(p,index=False)
            with self.assertRaises(ValueError):data.read_prefix(p)

    def test_strict_annual_maturity_with_synthetic_ledger(self):
        dates=pd.date_range('2020-01-01','2024-07-01')
        d=pd.DataFrame({'index':np.arange(len(dates))+21,'date':dates.strftime('%Y-%m-%d'),
            'label_end':(dates+np.timedelta64(5,'D')).strftime('%Y-%m-%d'),
            'direction':np.where(np.arange(len(dates))%17==0,0,np.where(np.arange(len(dates))%2==0,1,-1))})
        for year in [2022,2023,2024]:
            tr,va,ev,allva=data.split_data(d,year,strict=False)
            self.assertEqual(len(allva),126)
            self.assertTrue((d.loc[tr,'label_end']<d.loc[allva,'date'].min()).all())
            self.assertTrue((d.loc[allva,'label_end']<d.loc[ev,'date'].min()).all())
            self.assertTrue((d.loc[ev,'label_end']<=f'{year}-12-31').all())
            self.assertFalse(set(tr)&set(allva));self.assertFalse(d.loc[tr,'direction'].eq(0).any())


class PCATests(unittest.TestCase):
    def setUp(self):
        self.raw=np.random.default_rng(17).normal(size=(100,5))
        self.d=pd.DataFrame({'index':[21,22,30,70], 'direction':[1,-1,1,1]})
        self.tr=np.array([0,1,2]);self.pca=data.fit_pca(self.raw,self.d,self.tr)

    def test_unique_training_days_and_train_only_fit(self):
        expected=np.unique(np.concatenate([np.arange(1,21),np.arange(2,22),np.arange(10,30)]))
        self.assertEqual(self.pca['unique_day_indices'],expected.tolist())
        np.testing.assert_allclose(self.pca['mean'],self.raw[expected][:,data.EXTERNAL].mean(0))
        changed=self.raw.copy();changed[30:]*=1000;changed[:,3]+=12345
        self.assertEqual(self.pca,data.fit_pca(changed,self.d,self.tr))
        self.assertEqual(json.loads(json.dumps(self.pca)),self.pca)
        for v in np.array(self.pca['loadings']).T:self.assertGreaterEqual(v[np.argmax(np.abs(v))],0)
        np.testing.assert_allclose(np.array(self.pca['loadings']).T@np.array(self.pca['loadings']),np.eye(3),atol=1e-12)

    def test_features_shapes_and_order(self):
        x=np.stack([self.raw[1:21],self.raw[2:22]])
        spreads=np.arange(8).reshape(2,4)
        own=data.features('F0_Own20Logit',x,spreads)
        full=data.features('F1_FinancialLogit',x,spreads)
        pc=data.features('F2_PCALogit',x,spreads,self.pca)
        self.assertEqual(own.shape,(2,20));self.assertEqual(full.shape,(2,104));self.assertEqual(pc.shape,(2,84))
        np.testing.assert_equal(full[:,:20],x[:,:,3])
        np.testing.assert_equal(full[:,20:40],x[:,:,0]);np.testing.assert_equal(full[:,-4:],spreads)
        transformed=data.transform_pca(x,self.pca)
        self.assertEqual(transformed.shape,(2,20,3));np.testing.assert_equal(pc[:,20:40],transformed[:,:,0])
        bad=x.copy();bad[:,:,data.EXTERNAL]+=999
        np.testing.assert_equal(own,data.features('F0',bad,spreads))

    def test_transform_does_not_mutate_fitted_state(self):
        before=json.dumps(self.pca,sort_keys=True)
        x=np.stack([self.raw[1:21]])*1e8
        data.transform_pca(x,self.pca)
        self.assertEqual(before,json.dumps(self.pca,sort_keys=True))

    def test_constant_external_features_and_invalid_train(self):
        p=data.fit_pca(np.ones((100,5)),self.d,self.tr)
        self.assertTrue(p['zero_total_variance'])
        self.assertEqual(p['explained_variance_ratio'],[0.,0.,0.])
        with self.assertRaises(ValueError):data.fit_pca(self.raw,self.d,np.array([],int))
        with self.assertRaises(ValueError):data.fit_pca(self.raw,self.d,np.array([0,0]))
        d=self.d.copy();d.loc[1,'direction']=0
        with self.assertRaises(ValueError):data.fit_pca(self.raw,d,self.tr)


if __name__=='__main__':unittest.main()
