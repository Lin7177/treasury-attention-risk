"""Synthetic-only structural guards for the frozen target-centric implementation."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models_target import build_model,parameter_counts,NAMES,TARGET,OTHERS,fixed_mixer_seed


class TargetCentricTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)
        self.x=torch.randn(7,20,5)
        self.seed=202611041

    def test_shape_finite_backward_and_exact_parameter_counts(self):
        for name in NAMES:
            with self.subTest(name=name):
                model=build_model(name,self.seed)
                result=model(self.x)
                self.assertEqual(tuple(result.shape),(7,))
                self.assertTrue(torch.isfinite(result).all())
                result.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                expected=2652 if name.startswith(('N0','N1')) else 2657
                self.assertEqual(parameter_counts(model)['total'],expected)

    def test_initialization_pairs_and_common_backbone_bitwise(self):
        models=[build_model(name,self.seed) for name in NAMES]
        for a,b in [(models[0],models[1]),(models[2],models[3]),(models[3],models[4])]:
            self.assertEqual(set(a.state_dict()),set(b.state_dict()))
            for name,value in a.state_dict().items():self.assertTrue(torch.equal(value,b.state_dict()[name]),name)
        for model in models[1:]:
            for name,value in models[0].state_dict().items():
                if not name.startswith('mixer.'):
                    self.assertTrue(torch.equal(value,model.state_dict()[name]),name)

    def test_initialization_isolated_rng_and_stable_hash_seed(self):
        torch.manual_seed(222);before=torch.random.get_rng_state().clone()
        a=build_model('N1',self.seed)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        b=build_model('N1',self.seed)
        for key,value in a.state_dict().items():self.assertTrue(torch.equal(value,b.state_dict()[key]))
        self.assertEqual(fixed_mixer_seed(self.seed),fixed_mixer_seed(self.seed))
        self.assertNotEqual(fixed_mixer_seed(self.seed),fixed_mixer_seed(self.seed+1))

    def test_masked_own_only_external_invariance_and_gradient(self):
        model=build_model('N0',self.seed)
        perturbed=self.x.clone();perturbed[:,:,OTHERS]=torch.randn(7,20,4)*100
        self.assertTrue(torch.equal(model(self.x),model(perturbed)))
        perturbed[:,:,OTHERS]=float('nan')
        self.assertTrue(torch.equal(model(self.x),model(perturbed)))
        x=self.x.clone().requires_grad_();model(x).sum().backward()
        self.assertEqual(torch.count_nonzero(x.grad[:,:,OTHERS]).item(),0)
        self.assertGreater(torch.count_nonzero(x.grad[:,:,TARGET]).item(),0)

    def test_content_query_and_external_key_shape(self):
        model=build_model('N3',self.seed)
        logits,w=model(self.x,return_attention=True)
        self.assertEqual(tuple(w.shape),(7,2,1,4))
        torch.testing.assert_close(w.sum(-1),torch.ones(7,2,1))
        self.assertEqual(tuple(OTHERS),(0,1,2,4));self.assertNotIn(TARGET,OTHERS)
        logits.square().mean().backward()
        self.assertGreater(torch.count_nonzero(model.mixer.qkv.weight.grad[:32]).item(),0)

    def test_uniform_independent_of_qk_and_zero_task_gradient(self):
        model=build_model('N2',self.seed)
        before,w=model(self.x,return_attention=True)
        self.assertTrue(torch.equal(w,torch.full_like(w,.25)))
        with torch.no_grad():
            model.mixer.qkv.weight[:32].normal_(100,20)
            model.mixer.qkv.bias[:32].fill_(100.)
        after,other=model(self.x,return_attention=True)
        self.assertTrue(torch.equal(before,after));self.assertTrue(torch.equal(w,other))
        after.sum().backward()
        self.assertEqual(torch.count_nonzero(model.mixer.qkv.weight.grad[:32]).item(),0)
        self.assertEqual(torch.count_nonzero(model.mixer.qkv.bias.grad[:32]).item(),0)
        self.assertGreater(torch.count_nonzero(model.mixer.qkv.weight.grad[32:]).item(),0)
        self.assertEqual(parameter_counts(model)['structurally_zero_gradient_qk'],544)

    def test_all_variants_have_identical_explicit_target_residual(self):
        for name in NAMES:
            model=build_model(name,self.seed)
            with torch.no_grad():
                if name.startswith(('N0','N1')):
                    model.mixer[-1].weight.zero_();model.mixer[-1].bias.zero_()
                else:
                    model.mixer.output.weight.zero_();model.mixer.output.bias.zero_()
            h=model.encode(self.x)[:,TARGET]
            expected=model.readout(h+model.ffn(model.norm2(h))).squeeze(-1)
            torch.testing.assert_close(model(self.x),expected,atol=0,rtol=0)

    def test_n4_equals_full_self_target_and_canonical_old_initialization(self):
        # Read-only import of self-written old model source, not its checkpoint.
        parent=Path(__file__).resolve().parents[1]
        reference=parent/'v1'
        sys.path.insert(0,str(reference))
        from models import build_model as old_build
        torch.manual_seed(self.seed);canonical=old_build('T1_attention')
        target=build_model('N4',self.seed)
        self.assertEqual(set(canonical.state_dict()),set(target.state_dict()))
        for key,value in canonical.state_dict().items():
            self.assertTrue(torch.equal(value,target.state_dict()[key]),key)
        tokens=canonical.encode(self.x)
        full=canonical.mix_tokens(tokens)
        expected=canonical.readout(full[:,TARGET]).squeeze(-1)
        actual,w=target(self.x,return_attention=True)
        self.assertEqual(tuple(w.shape),(7,2,1,5))
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
        self.assertFalse(torch.allclose(actual,canonical(self.x),atol=1e-6,rtol=0))

    def test_fixed_mixer_and_norm_path_exactly_specified(self):
        model=build_model('N1',self.seed)
        self.assertEqual((model.mixer[0].in_features,model.mixer[0].out_features),(80,11))
        self.assertEqual((model.mixer[2].in_features,model.mixer[2].out_features),(11,16))
        h=model.encode(self.x);context=model.mixer(model.norm1(h).flatten(1))
        u=h[:,TARGET]+context
        torch.testing.assert_close(model(self.x),model.readout(u+model.ffn(model.norm2(u))).squeeze(-1),atol=0,rtol=0)

    def test_inputs_reject_state_and_wrong_history_shapes(self):
        with self.assertRaises(ValueError):build_model('unknown',self.seed)
        with self.assertRaises(ValueError):build_model('N3',self.seed)(self.x[:,:19])
        with self.assertRaises(ValueError):build_model('N3',self.seed)(self.x[:,:,:3])


if __name__=='__main__':unittest.main()
