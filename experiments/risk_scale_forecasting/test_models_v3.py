"""Synthetic-only guards for models_v3; no market data, optimizer, or fit."""
import copy
import math
import unittest

import torch
from torch.nn import functional as F

from models_v3 import (
    MODEL_NAMES, NORMALIZED_FLOOR, SHARED_COMPONENTS, build_model, subseed,
)


class ModelGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def x(self, batch=7):
        return torch.randn(batch, 20, 10, generator=torch.Generator().manual_seed(91027))

    def test_parameter_and_shared_initialization(self):
        models = [build_model(n, 202611051) for n in MODEL_NAMES]
        self.assertEqual([m.parameter_summary()["total"] for m in models], [915, 857, 857])
        self.assertEqual([m.parameter_summary()["inactive_qk"] for m in models], [0, 144, 0])
        for key, value in models[0].state_dict().items():
            if key.split(".")[0] in SHARED_COMPONENTS:
                for other in models[1:]:
                    self.assertTrue(torch.equal(value, other.state_dict()[key]), key)
        self.assertEqual(len({m.initialization_hashes()["shared"] for m in models}), 1)
        self.assertEqual(models[1].initialization_hashes()["mixer"], models[2].initialization_hashes()["mixer"])
        self.assertEqual(sum(t.numel() for k, t in models[0].state_dict().items()
                             if k.split(".")[0] in SHARED_COMPONENTS), 569)

    def test_constructor_rng_and_shuffle_stream_isolation(self):
        torch.manual_seed(101)
        before = torch.get_rng_state().clone()
        shuffle = torch.Generator().manual_seed(202611051)
        expected = torch.randperm(101, generator=shuffle)
        for name in ("Content", "Fixed", "Uniform", "Content"):
            build_model(name, 202611051)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
        shuffle = torch.Generator().manual_seed(202611051)
        self.assertTrue(torch.equal(expected, torch.randperm(101, generator=shuffle)))
        self.assertNotEqual(subseed(12, "time_fixed"), subseed(12, "time_attention"))
        self.assertEqual(subseed(12, "time_fixed"), subseed(12, "time_fixed"))
        self.assertEqual(build_model("Fixed", 12).initialization_hashes(),
                         build_model("Fixed", 12).initialization_hashes())
        self.assertNotEqual(build_model("Fixed", 12).initialization_hashes()["mixer"],
                            build_model("Fixed", 13).initialization_hashes()["mixer"])

    def test_output_raw_positive_zero_target_and_batch_one(self):
        for name in MODEL_NAMES:
            m = build_model(name, 202611051)
            for x in (torch.zeros(1, 20, 10), self.x(), self.x() * 1e3):
                u, raw = m(x, return_raw=True)
                self.assertEqual(u.shape, (x.shape[0],))
                self.assertTrue(torch.isfinite(u).all())
                self.assertTrue((raw >= 0).all())
                self.assertTrue((u >= NORMALIZED_FLOOR).all())
                if x.abs().max() < 10:
                    self.assertTrue((raw > NORMALIZED_FLOOR).all())
                self.assertTrue(torch.allclose(u, raw + NORMALIZED_FLOOR, atol=0, rtol=0))
                # q=0 is legal; its loss and task gradients must remain finite.
                loss = torch.log(u).mean()
                m.zero_grad()
                loss.backward()
                for p in m.parameters():
                    if p.grad is not None:
                        self.assertTrue(torch.isfinite(p.grad).all())

    def test_twenty_time_kv_latest_query_and_attention_shape(self):
        m = build_model("Content", 202611051)
        x = self.x()
        u, weights = m(x, return_attention=True)
        self.assertEqual(tuple(weights.shape), (7, 2, 1, 20))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(7, 2, 1), atol=1e-6))
        a = m.norm1(m.encode(x))
        wq, wk, _ = m.attention.in_proj_weight.chunk(3)
        bq, bk, _ = m.attention.in_proj_bias.chunk(3)
        q = F.linear(a[:, -1:], wq, bq).reshape(7, 1, 2, 4).transpose(1, 2)
        k = F.linear(a, wk, bk).reshape(7, 20, 2, 4).transpose(1, 2)
        expected = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(4), -1)
        self.assertTrue(torch.allclose(weights, expected, atol=1e-7, rtol=0))
        changed = x.clone()
        changed[:, -1, :] += torch.linspace(-3, 3, 10)
        self.assertFalse(torch.allclose(weights, m(changed, return_attention=True)[1]))
        self.assertTrue(torch.isfinite(u).all())

    def test_uniform_dead_qk_and_content_live_qk_task_gradients(self):
        for name in ("Uniform", "Content"):
            m = build_model(name, 202611051)
            u = m(self.x())
            q = torch.linspace(0, 3, len(u))
            (torch.log(u) + q / u).mean().backward()
            weight_grad = m.attention.in_proj_weight.grad
            bias_grad = m.attention.in_proj_bias.grad
            self.assertIsNotNone(weight_grad)
            self.assertIsNotNone(bias_grad)
            qk = torch.cat((weight_grad[:16].flatten(), bias_grad[:16]))
            self.assertEqual(qk.numel(), 144)
            self.assertTrue(torch.isfinite(qk).all())
            if name == "Uniform":
                self.assertEqual(torch.count_nonzero(qk).item(), 0)
                # Retained Q/K values do not alter the task function.
                reference = m(self.x()).detach()
                with torch.no_grad():
                    m.attention.in_proj_weight[:16].add_(33)
                    m.attention.in_proj_bias[:16].add_(33)
                self.assertTrue(torch.equal(reference, m(self.x())))
            else:
                self.assertGreater(torch.count_nonzero(weight_grad[:8]).item(), 0)
                self.assertGreater(torch.count_nonzero(weight_grad[8:16]).item(), 0)
                self.assertGreater(torch.count_nonzero(qk).item(), 0)
            self.assertGreater(torch.count_nonzero(weight_grad[16:]).item(), 0)
            self.assertGreater(torch.count_nonzero(m.attention.out_proj.weight.grad).item(), 0)

    def test_same_weight_uniform_counterfactual(self):
        content = build_model("Content", 202611051)
        uniform = build_model("Uniform", 202611051)
        uniform.load_state_dict(copy.deepcopy(content.state_dict()))
        forced, fw = content(self.x(), force_uniform=True, return_attention=True)
        actual, uw = uniform(self.x(), return_attention=True)
        self.assertTrue(torch.allclose(forced, actual, atol=1e-6, rtol=0))
        self.assertTrue(torch.equal(fw, torch.full((7, 2, 1, 20), 1 / 20)))
        self.assertTrue(torch.equal(fw, uw))
        # Cross-check against the framework full Self-Attention latest-row output.
        h = content.encode(self.x())
        a = content.norm1(h)
        full_context, _ = content.attention(a, a, a, need_weights=False)
        our_context, _ = content.aggregate(h)
        self.assertTrue(torch.allclose(full_context[:, -1], our_context, atol=1e-6, rtol=0))

    def test_shared_recent_residual_and_other_history_paths(self):
        x = self.x()
        for name in MODEL_NAMES:
            m = build_model(name, 202611051)
            # With every context parameter zero, all models share exact recent path.
            with torch.no_grad():
                if name == "Fixed":
                    for p in (m.fixed1, m.fixed2):
                        for value in p.parameters():
                            value.zero_()
                else:
                    for p in m.attention.parameters():
                        p.zero_()
            h = m.encode(x)
            recent = h[:, -1]
            v = recent + m.ffn2(F.gelu(m.ffn1(m.norm2(recent))))
            expected = F.softplus(m.head(v).squeeze(-1)) + NORMALIZED_FLOOR
            self.assertTrue(torch.equal(m(x), expected))
            altered = x.clone()
            altered[:, -1] += 0.5
            self.assertFalse(torch.allclose(m(x), m(altered)))
        for name in MODEL_NAMES:
            m = build_model(name, 202611051)
            altered = x.clone()
            altered[:, 0] += torch.linspace(-5, 5, 10)
            self.assertFalse(torch.allclose(m(x), m(altered), atol=1e-8, rtol=0))
        fixed = build_model("Fixed", 202611051)
        self.assertFalse(torch.allclose(fixed(x), fixed(x.flip(1)), atol=1e-8, rtol=0))

    def test_unit_qlike_constant_and_pair_difference(self):
        q = torch.tensor([0, .01, 1, 30], dtype=torch.float64)
        u = torch.tensor([.1, .3, 1, 10], dtype=torch.float64)
        v = torch.tensor([.2, .2, 2, 20], dtype=torch.float64)
        scale = 123.0
        norm = torch.log(u) + q / u
        physical = torch.log(scale * u) + scale * q / (scale * u)
        self.assertTrue(torch.allclose(physical - norm, torch.full_like(q, math.log(scale)), atol=1e-13))
        pair1 = torch.log(u) - torch.log(v) + q * (1 / u - 1 / v)
        pair2 = torch.log(scale * u) - torch.log(scale * v) + scale * q * (1 / (scale * u) - 1 / (scale * v))
        self.assertTrue(torch.allclose(pair1, pair2, atol=1e-13))

    def test_replay_determinism_input_guards_and_raw_floor_exposure(self):
        x = self.x()
        for name in MODEL_NAMES:
            m = build_model(name, 202611051).eval()
            first = m(x).detach()
            rebuilt = build_model(name, 19)
            rebuilt.load_state_dict(copy.deepcopy(m.state_dict()))
            self.assertTrue(torch.equal(first, rebuilt.eval()(x)))
            self.assertTrue(torch.equal(first, m(x)))
            for bad in (torch.zeros(20, 10), torch.zeros(2, 19, 10), torch.zeros(0, 20, 10),
                        torch.full((1, 20, 10), float("nan"))):
                with self.assertRaises(ValueError):
                    m(bad)
            with torch.no_grad():
                m.head.weight.zero_()
                m.head.bias.fill_(-100)
            u, raw = m(x, return_raw=True)
            self.assertTrue((raw <= NORMALIZED_FLOOR).all())
            self.assertTrue((u >= NORMALIZED_FLOOR).all())
            # Runner can detect failure instead of the stabilized u hiding it.


if __name__ == "__main__":
    unittest.main()
