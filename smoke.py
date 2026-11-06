"""Exercise research components on generated inputs only."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CHECKS = {
    "v1": """
import torch
from models import build_model
torch.manual_seed(712)
x = torch.randn(12, 20, 5)
for name in ('T1_attention', 'B4_uniform'):
    model = build_model(name)
    loss = model(x).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
""",
    "v2": """
import torch
from models_target import build_model
x = torch.randn(12, 20, 5, generator=torch.Generator().manual_seed(712))
for name in ('N1', 'N2', 'N3', 'N4'):
    assert torch.isfinite(build_model(name, 712)(x)).all()
""",
    "v3": """
import numpy as np
import torch
from models_v3 import build_model
from numerical_v3 import fit_har, predict_har, qlike
rng = np.random.default_rng(712)
x = rng.normal(size=(64, 20, 5))
y = rng.uniform(.2, 2, size=64)
z = np.c_[np.ones(64), x[:, -1, :2]]
state = fit_har(z[:48], y[:48])
assert np.isfinite(qlike(y[48:], predict_har(state, z[48:]))).all()
xt = torch.tensor(np.concatenate([x, x*x], axis=-1), dtype=torch.float32)
for name in ('Fixed', 'Uniform', 'Content'):
    assert torch.isfinite(build_model(name, 712)(xt)).all()
""",
    "v4": """
import numpy as np
import torch
from financial_v4 import dcc_filter, matrix_qlike
from models_v4 import build_model, fixed_matrix, kernel_bandwidth_statistics
x = torch.randn(12, 20, 5, generator=torch.Generator().manual_seed(712), dtype=torch.float64)
prior = torch.eye(3, dtype=torch.float64)
dcc = dcc_filter(x[:3, :, :3].reshape(-1, 3).numpy(), np.eye(3), .04, .9)
assert np.isfinite(dcc['loss'])
band = kernel_bandwidth_statistics(x[:8])
with torch.no_grad():
    for name in ('Static', 'Content', 'MLP'):
        cov = build_model(name, 712)(x[8:], prior).numpy()
        assert np.isfinite(matrix_qlike(np.broadcast_to(np.eye(3), cov.shape), cov)).all()
    assert torch.isfinite(fixed_matrix(x[8:], prior, 'Uniform')).all()
assert band
""",
    "v5": """
import json
from pathlib import Path
import sys
import numpy as np
import torch
from hedge_v5 import Scorer, action, components, risk, rule
from make_paper_v5 import plot
cfg = json.loads(Path('frozen_config.json').read_text(encoding='utf-8'))
rng = torch.Generator().manual_seed(712)
q = torch.randn(8, 6, generator=rng, dtype=torch.float64)
k = torch.randn(8, 12, 6, generator=rng, dtype=torch.float64)
v = torch.rand(8, 12, 2, generator=rng, dtype=torch.float64)
mask = torch.ones(8, 12, dtype=torch.bool)
c = components(np.broadcast_to(np.eye(3), (8, 3, 3)))
u0 = torch.as_tensor(action(*c[:2]))
scorer = Scorer('Content', 712)
for u in (scorer(q,k,v,mask,u0,cfg), scorer(q,k,v,mask,u0,cfg,no_query=True),
          rule('MemoryUniform',q,k,v,mask,u0,cfg)):
    assert np.isfinite(risk(c, u.detach().numpy())).all()
out = Path(sys.argv[1])
# Arbitrary interval placeholders for testing the renderer only.
pairs = [{'market': market, 'baseline': name, 'block': 10,
          'point': 0., 'lower': -.1, 'upper': .1}
         for market in ('us', 'china') for name in
         ('DCC','Kernel','MemoryKernel','MemoryUniform','MLP','ContentNoQuery','Ridge')]
summary = out / 'synthetic_plot_fixture.json'
summary.write_text(json.dumps({'evidence': 'synthetic_only', 'pairs': pairs}), encoding='utf-8')
plot(summary, out / 'SYNTHETIC_pairs.png', synthetic=True)
""",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for version, check in CHECKS.items():
        subprocess.run(
            [sys.executable, "-B", "-c", "import torch; torch.set_num_threads(1)\n" + check, str(output)],
            cwd=ROOT / version,
            check=True,
        )
    (output / "status.json").write_text(
        json.dumps({"evidence": "synthetic_only", "passed": list(CHECKS)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Synthetic checks passed for V1–V5. No historical experiment was run.")


if __name__ == "__main__":
    main()
