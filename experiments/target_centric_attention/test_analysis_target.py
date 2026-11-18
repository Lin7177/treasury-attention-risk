import unittest
import numpy as np
from analyze_target import metrics,macro_ba,joint_block_indices,matrix_statistics,paired_bootstrap

class FrozenAnalysisTests(unittest.TestCase):
    def test_flat_counts_wrong_but_not_in_binary_metrics(self):
        r=metrics([.9,.1,.9],[1,-1,0])
        self.assertAlmostEqual(r['accuracy'],2/3);self.assertEqual(r['balanced_accuracy'],1.)
        self.assertAlmostEqual(r['brier'],.01);self.assertEqual(r['n'],3);self.assertEqual(r['nonflat_n'],2)
    def test_missing_class_and_all_flat_are_undefined(self):
        self.assertTrue(np.isnan(metrics([.2,.3],[-1,-1])['balanced_accuracy']))
        self.assertTrue(np.isnan(metrics([.2,.3],[0,0])['brier']))
        self.assertTrue(np.isnan(macro_ba([.2,.3],[-1,-1],['2022','2023'],('2022','2023'))))
    def test_macro_and_pooled_differ_for_year_constant_directions(self):
        y=np.array([1]*9+[-1]+[1]+[-1]*9);p=np.array([.9]*10+[.1]*10);years=np.array(['2022']*10+['2023']*10)
        self.assertAlmostEqual(metrics(p,y)['balanced_accuracy'],.9)
        self.assertAlmostEqual(macro_ba(p,y,years,('2022','2023')),.5)
    def test_joint_draw_preserves_folds_and_exact_pair_identity(self):
        years=np.array(['2022']*5+['2023']*7)
        ids=joint_block_indices(years,3,np.random.default_rng(11))
        self.assertEqual(len(ids),12);self.assertTrue((years[ids[:5]]=='2022').all());self.assertTrue((years[ids[5:]]=='2023').all())
        values=np.arange(12);self.assertTrue(np.array_equal(values[ids],values.copy()[ids]))
    def test_identical_models_have_zero_intervals(self):
        y=np.array([1,-1]*12);years=np.array(['2022']*12+['2023']*12);p=np.linspace(.1,.9,24);matrix=np.c_[p,p]
        result=paired_bootstrap(matrix,y,years,['a','b'],pairs=(('a','b'),),repetitions=30,block_lengths=(2,),minimum_valid=20,expected_years=('2022','2023'))
        self.assertTrue(result.low.eq(0).all());self.assertTrue(result.high.eq(0).all());self.assertTrue(result.valid_enough.all())
    def test_missing_class_replicates_not_replaced_or_redrawn(self):
        y=np.array([1,1,-1,-1]);years=np.array(['2022','2022','2023','2023']);p=np.c_[np.ones(4)*.6,np.ones(4)*.4]
        result=paired_bootstrap(p,y,years,['a','b'],pairs=(('a','b'),),repetitions=25,block_lengths=(1,),minimum_valid=20,expected_years=('2022','2023'))
        q=result[result.metric=='macro_balanced_accuracy'].iloc[0]
        self.assertEqual(q.valid_replicates,0);self.assertFalse(q.valid_enough);self.assertTrue(np.isnan(q.low));self.assertEqual(q.requested_replicates,25)
    def test_matrix_metric_matches_scalar(self):
        p=np.array([.7,.2,.3,.8,.1,.9]);y=np.array([1,-1,0,1,-1,0]);years=np.array(['2022']*3+['2023']*3)
        result=matrix_statistics(p[:,None],y,years,('2022','2023'))
        self.assertAlmostEqual(result['pooled_balanced_accuracy'][0],metrics(p,y)['balanced_accuracy'])
        self.assertAlmostEqual(result['macro_balanced_accuracy'][0],macro_ba(p,y,years,('2022','2023')))

class AdditionalAnalysisGuards(unittest.TestCase):
    def test_brier_uses_actual_probabilities_not_clipped_values(self):
        self.assertEqual(metrics([1.,0.],[1,-1])['brier'],0.)
    def test_primary_only_gets_conservative_macro_interval(self):
        y=np.array([1,-1]*6);years=np.array(['2022']*6+['2023']*6)
        probabilities=np.c_[np.linspace(.1,.9,12),np.linspace(.2,.8,12),np.linspace(.3,.7,12)]
        names=['N3_ContentCross','N2_UniformCross','other']
        pairs=(('N3_ContentCross','N2_UniformCross'),('N3_ContentCross','other'))
        result=paired_bootstrap(probabilities,y,years,names,pairs=pairs,repetitions=10,block_lengths=(1,),minimum_valid=0,expected_years=('2022','2023'))
        conservative=result[result.confidence_level==.975]
        self.assertEqual(len(conservative),1)
        self.assertEqual(conservative.metric.iloc[0],'macro_balanced_accuracy')
        self.assertEqual(conservative.right.iloc[0],'N2_UniformCross')
    def test_bootstrap_deterministic_seed(self):
        y=np.array([1,-1]*6);years=np.array(['2022']*6+['2023']*6);p=np.c_[np.linspace(.1,.9,12),np.ones(12)*.6]
        a=paired_bootstrap(p,y,years,['a','b'],pairs=(('a','b'),),repetitions=12,block_lengths=(2,),minimum_valid=0,expected_years=('2022','2023'))
        b=paired_bootstrap(p,y,years,['a','b'],pairs=(('a','b'),),repetitions=12,block_lengths=(2,),minimum_valid=0,expected_years=('2022','2023'))
        self.assertTrue(a.equals(b))

if __name__=='__main__':unittest.main()
