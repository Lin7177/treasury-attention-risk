import unittest
import torch
from models import MODEL_NAMES, build_model, parameter_counts


class BoundaryModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(4,20,5)
        self.state = torch.randn(4,3)

    def test_shapes_and_finite_backward_all_models(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model=build_model(name)
                logits=model(self.x,self.state)
                self.assertEqual(tuple(logits.shape),(4,))
                self.assertTrue(torch.isfinite(logits).all())
                logits.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_seed_reproducibility(self):
        torch.manual_seed(99);a=build_model('T1_attention')
        torch.manual_seed(99);b=build_model('T1_attention')
        self.assertTrue(torch.equal(a(self.x,self.state),b(self.x,self.state)))

    def test_uniform_qk_are_dead_not_capacity_matched(self):
        model=build_model('B4_uniform');model(self.x,self.state).sum().backward()
        dim=model.dim
        self.assertEqual(torch.count_nonzero(model.mixer.qkv.weight.grad[:2*dim]).item(),0)
        self.assertEqual(torch.count_nonzero(model.mixer.qkv.bias.grad[:2*dim]).item(),0)
        self.assertGreater(torch.count_nonzero(model.mixer.qkv.weight.grad[2*dim:]).item(),0)
        self.assertEqual(parameter_counts(model)['structurally_zero_gradient_qk'],544)
        self.assertEqual(parameter_counts(model)['total'],parameter_counts(build_model('T1_attention'))['total'])

    def test_token_mixer_is_permutation_equivariant(self):
        for name in ['T1_attention','B4_uniform']:
            model=build_model(name);tokens=model.encode(self.x);order=torch.tensor([4,2,0,1,3])
            torch.testing.assert_close(model.mix_tokens(tokens[:,order]),model.mix_tokens(tokens)[:,order],atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(model.mix_tokens(tokens[:,order]).mean(1),model.mix_tokens(tokens).mean(1),atol=1e-6,rtol=1e-5)

    def test_tenor_identity_and_history_order_not_removed(self):
        model=build_model('T1_attention')
        self.assertFalse(torch.allclose(model(self.x,self.state),model(self.x.flip(2),self.state)))
        self.assertFalse(torch.allclose(model(self.x,self.state),model(self.x.flip(1),self.state)))

    def test_film_initial_identity_matches_poststate(self):
        torch.manual_seed(13);condition=build_model('T2_condition')
        torch.manual_seed(13);post=build_model('T3_post_state')
        torch.testing.assert_close(condition(self.x,self.state),post(self.x,self.state),atol=0,rtol=0)
        condition(self.x,self.state).sum().backward()
        self.assertGreater(torch.count_nonzero(condition.film.weight.grad).item(),0)

    def test_state_enters_condition_but_not_unconditioned(self):
        model=build_model('T1_attention')
        torch.testing.assert_close(model(self.x,self.state),model(self.x,self.state+10),atol=0,rtol=0)
        conditioned=build_model('T2_condition')
        with torch.no_grad():conditioned.film.weight.fill_(.05)
        a=conditioned.mix_tokens(conditioned.encode(self.x))
        gamma,beta=conditioned.film(self.state).chunk(2,-1)
        b=conditioned.mix_tokens(conditioned.encode(self.x)*(1+gamma[:,None])+beta[:,None])
        self.assertFalse(torch.allclose(a,b))

    def test_state_changes_attention_weights_only_in_conditioned_model(self):
        def weights(model,x,state):
            tokens=model.encode(x)
            if model.film is not None:
                gamma,beta=model.film(state).chunk(2,-1)
                tokens=tokens*(1+gamma[:,None])+beta[:,None]
            tokens=model.norm1(tokens)
            b,length,dim=tokens.shape
            qkv=model.mixer.qkv(tokens).reshape(b,length,3,2,dim//2)
            q,k,_=qkv.permute(2,0,3,1,4).unbind(0)
            return torch.softmax(q@k.transpose(-2,-1)/(dim//2)**.5,-1)
        post=build_model('T3_post_state')
        torch.testing.assert_close(weights(post,self.x,self.state),weights(post,self.x,self.state+1),atol=0,rtol=0)
        conditioned=build_model('T2_condition')
        with torch.no_grad():
            torch.manual_seed(17)
            conditioned.film.weight.copy_(torch.randn_like(conditioned.film.weight)*.15)
        self.assertFalse(torch.allclose(weights(conditioned,self.x,self.state),weights(conditioned,self.x,self.state+1)))

    def test_shape_and_name_errors(self):
        with self.assertRaises(ValueError):build_model('unknown')
        with self.assertRaises(ValueError):build_model('T1_attention')(self.x[:,:19],self.state)
        with self.assertRaises(ValueError):build_model('T2_condition')(self.x,torch.ones(4,2))


if __name__=='__main__':unittest.main()
