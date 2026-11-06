"""Independent V5 checks using only hand-built synthetic fixtures."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import data_v5
import hedge_v5 as hedge


class HedgeTests(unittest.TestCase):
    def test_risk_matches_direct_quadratic_form_and_minimizer(self):
        h = np.array([[[4., 1., 0.], [1., 3., 0.], [0., 0., 2.]]])
        c = hedge.components(h)
        u = hedge.action(*c[:2])
        np.testing.assert_allclose(u, [.5])
        exposure = hedge.B - u[:, None] * hedge.A
        np.testing.assert_allclose(
            hedge.risk(c, u), np.einsum("ni,nij,nj->n", exposure, h, exposure)
        )
        self.assertLess(hedge.risk(c, u).item(), hedge.risk(c, np.zeros(1)).item())

    def test_action_fallback_bounds_and_local_cap(self):
        np.testing.assert_allclose(
            hedge.action(np.array([0., 1., 1.]), np.array([9., -1., 2.])),
            [.6, 0., 1.],
        )
        np.testing.assert_allclose(
            hedge.local(np.array([.05, .5, .95]), np.array([-1., 1., 1.]), .2),
            [0., .7, 1.],
        )

    def test_invalid_risk_rejected(self):
        for yy in (-1., float("nan")):
            with self.subTest(yy=yy), self.assertRaisesRegex(ValueError, "Invalid decision risk"):
                hedge.risk((np.ones(1), np.zeros(1), np.array([yy])), np.zeros(1))

    def test_no_query_removes_query_but_keeps_recency_and_mask(self):
        model = hedge.Scorer("Content", 712)
        with torch.no_grad():
            model.query.zero_()
            model.key.zero_()
            model.query[0, 0] = 1.
            model.key[0, 0] = 1.
            model.age.fill_(.5)
        q = torch.zeros((2, 6), dtype=torch.float64)
        q[:, 0] = torch.tensor([4., -4.])
        k = torch.zeros((2, 3, 6), dtype=torch.float64)
        k[:, :, 0] = torch.tensor([1., -1., 100.])
        v = torch.tensor([[[.2, 1.], [.8, 1.], [999., 1.]]] * 2, dtype=torch.float64)
        mask = torch.tensor([[True, True, False]] * 2)
        anchor = torch.full((2,), .5, dtype=torch.float64)
        cfg = {"age_decay": .4, "local_penalty": .5, "local_cap": .5}
        off = model(q, k, v, mask, anchor, cfg, no_query=True)
        on = model(q, k, v, mask, anchor, cfg)
        weights = np.exp([0., -.6])
        expected = .5 + (np.dot(weights, [.2, .8]) / weights.sum() - .5) / 1.5
        np.testing.assert_allclose(off.detach().numpy(), [expected, expected])
        self.assertLess(on[0].item(), off[0].item())
        self.assertGreater(on[1].item(), off[1].item())


class ReplayGuardTests(unittest.TestCase):
    def test_memory_excludes_same_day_and_other_fold(self):
        df = pd.DataFrame({
            "market": ["synthetic"] * 5,
            "fold": [1, 1, 1, 1, 2],
            "date": pd.date_range("2000-01-01", periods=5),
            "label_end": pd.date_range("2000-01-02", periods=5),
        })
        history = hedge.history_positions(df, memory=1)
        self.assertEqual([r.tolist() for r in history], [[], [], [0], [1], []])
        np.testing.assert_array_equal(data_v5.matured_positions(df, 3), [0, 1])

    def test_private_loader_rejects_even_existing_path_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text("{}", encoding="utf-8")
            with patch.object(Path, "open", side_effect=AssertionError("unexpected file read")):
                for market in ("us", "china"):
                    with self.subTest(market=market), self.assertRaisesRegex(RuntimeError, "authorized local"):
                        data_v5.load_market(path, market, roles=("evaluation",))

    def test_historical_runner_stops_before_loading_or_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(hedge, "ROOT", root), patch.object(
                hedge, "load_market", side_effect=AssertionError("unexpected data load")
            ), patch.object(
                hedge, "fit_model", side_effect=AssertionError("unexpected fit")
            ), patch.object(hedge.torch, "set_num_threads"), patch.object(
                hedge.torch, "use_deterministic_algorithms"
            ), self.assertRaisesRegex(FileNotFoundError, "FREEZE_MANIFEST"):
                hedge.run()
            self.assertFalse((root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
