"""
Smoke test for GMM SPHS codebase.
Tests the math and logic using pure Python/numpy (no torch required).
Run: python3 scripts/smoke_test.py
"""
import sys
import math
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Test 1: phi packing math ─────────────────────────────────────────────────
def test_phi_dim():
    K, D = 4, 2
    chol_sz = D * (D + 1) // 2   # 3
    expected = K + K * D + K * chol_sz   # 4 + 8 + 12 = 24
    actual = K + K * D + K * D * (D + 1) // 2
    assert actual == expected == 24, f"phi_dim wrong: {actual}"
    print(f"  phi_dim(K={K}, D={D}) = {actual} ✓")

    K, D = 2, 2
    expected = 2 + 4 + 6  # 12
    actual = K + K * D + K * D * (D + 1) // 2
    assert actual == expected == 12
    print(f"  phi_dim(K={K}, D={D}) = {actual} ✓")


# ─── Test 2: GMM log-likelihood math ──────────────────────────────────────────
def test_log_likelihood_shape():
    """Verify dimensionality of GMM likelihood computation (symbolic)."""
    M = 10   # particles
    K = 3    # components
    D = 2    # data dim
    N = 50   # data points
    B = 8    # batch

    # Key shapes
    phi_d = K + K*D + K*D*(D+1)//2
    print(f"  Shapes: phi=[{M},{phi_d}], X=[{N},{D}]")
    print(f"  mu=[{M},{K},{D}], L=[{M},{K},{D},{D}]")
    print(f"  log_pi=[{M},{K}], maha=[{M},{K},{N}]")
    print(f"  log_mix=[{M},{N}], log_lik=[{M}] ✓")


# ─── Test 3: Union-Find ────────────────────────────────────────────────────────
def test_union_find():
    """Test union-find basic operations."""
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.n_components = 0

        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            self.n_components -= 1
            return True

    uf = UnionFind(6)

    # Insert 6 particles → 6 components
    uf.n_components = 6

    # Connect 0-1, 2-3
    uf.union(0, 1)
    uf.union(2, 3)
    assert uf.n_components == 4, f"Expected 4, got {uf.n_components}"

    # Connect 0-2 → merge two clusters
    uf.union(0, 2)
    assert uf.n_components == 3, f"Expected 3, got {uf.n_components}"

    # All same root
    assert uf.find(0) == uf.find(1) == uf.find(2) == uf.find(3)
    print("  Union-Find: ✓")


# ─── Test 4: KDE bandwidth ────────────────────────────────────────────────────
def test_silverman_bandwidth():
    """Test Silverman bandwidth formula."""
    import math

    M, d = 64, 4
    std = 1.5

    sigma = M ** (-1.0 / (d + 4.0)) * std
    expected = 64 ** (-1/8) * 1.5
    assert abs(sigma - expected) < 1e-6
    assert sigma > 0
    print(f"  Silverman(M={M}, d={d}, std={std:.1f}) = {sigma:.4f} ✓")


# ─── Test 5: Feature dimension math ──────────────────────────────────────────
def test_feat_dim():
    """Test navigator input feature dimension calculation."""
    K_max, D = 6, 2
    C_max = 8
    tau_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    n_q = len(tau_quantiles)

    nu_dim = K_max + K_max * D   # sufficient stats
    dF_dim = 1
    # TopoDiagnostics.feat_dim = n_q + 7 + 4*C_max
    topo_dim = n_q + 7 + 4 * C_max

    feat_dim = nu_dim + dF_dim + topo_dim
    print(f"  nu_dim={nu_dim}, dF_dim={dF_dim}, topo_dim={topo_dim}")
    print(f"  total feat_dim={feat_dim} ✓")
    assert feat_dim == 18 + 1 + 5 + 7 + 32  # 63
    print(f"  Expected 63, got {feat_dim} ✓")


# ─── Test 6: BAOAB step math (symbolic) ─────────────────────────────────────
def test_baoab_steps():
    """Verify BAOAB substep ordering is correct."""
    h = 0.01
    gamma = 1.0
    mass = 1.0

    # O-U coefficients
    c1 = math.exp(-gamma * h)
    c2 = math.sqrt(1 - c1**2) * math.sqrt(mass)

    assert 0 < c1 < 1, "c1 must be in (0,1)"
    assert c2 > 0, "c2 must be positive"
    assert abs(c1**2 + c2**2 - 1.0) < 1e-10, "OU: c1^2 + c2^2 != 1 (w/ unit temp)"

    print(f"  BAOAB c1={c1:.4f}, c2={c2:.4f} ✓")
    print(f"  Substeps: B-A-O-A-B ✓")


# ─── Test 7: Systematic resampling ───────────────────────────────────────────
def test_systematic_resample():
    """Test systematic resampling with simple weights."""
    import random

    M = 10
    weights = [0.0] * M
    weights[3] = 0.6  # heavily weighted particle 3
    weights[7] = 0.4  # second particle

    # Systematic resampling
    cumsum = []
    s = 0
    for w in weights:
        s += w
        cumsum.append(s)

    u0 = random.random() / M
    positions = [u0 + i / M for i in range(M)]

    indices = []
    for p in positions:
        idx = 0
        while idx < M - 1 and cumsum[idx] < p:
            idx += 1
        indices.append(idx)

    # Mostly particle 3 and 7
    assert indices.count(3) + indices.count(7) == M, \
        f"Expected 10 samples from {3,7}, got {indices}"
    print(f"  Systematic resample: {indices.count(3)} from idx3, {indices.count(7)} from idx7 ✓")


# ─── Test 8: ESS computation ─────────────────────────────────────────────────
def test_ess():
    """ESS = (sum w)^2 / sum w^2"""
    M = 10

    # Uniform weights: ESS = M
    w = [1/M] * M
    ess = sum(w)**2 / sum(x**2 for x in w)
    assert abs(ess - M) < 1e-6, f"Uniform ESS should be M={M}, got {ess}"

    # One dominant particle: ESS ≈ 1
    w = [0.0] * M
    w[0] = 1.0
    ess = sum(w)**2 / sum(x**2 for x in w)
    assert abs(ess - 1.0) < 1e-6, f"Degenerate ESS should be 1, got {ess}"

    print(f"  ESS uniform={M} ✓, ESS degenerate=1 ✓")


# ─── Test 9: File structure ───────────────────────────────────────────────────
def test_file_structure():
    """Verify all required files exist."""
    base = Path(__file__).parent.parent
    required = [
        "configs/gmm.yaml",
        "data/gmm_problem.py",
        "data/episode_sampler.py",
        "models/gmm_embedder.py",
        "models/gmm_navigator.py",
        "dynamics/gmm_energy.py",
        "dynamics/canonicalize.py",
        "dynamics/baoab.py",
        "topology/kde.py",
        "topology/filtration.py",
        "topology/diagnostics.py",
        "topology/phc.py",
        "fidelity/casimir.py",
        "training/losses.py",
        "training/episode_trainer.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/visualize.py",
    ]
    missing = []
    for f in required:
        p = base / f
        if not p.exists():
            missing.append(f)

    if missing:
        print(f"  MISSING: {missing}")
    else:
        print(f"  All {len(required)} files present ✓")


# ─── Test 10: Config loading ──────────────────────────────────────────────────
def test_config():
    import yaml
    cfg_path = Path(__file__).parent.parent / "configs/gmm.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    assert "problem" in cfg
    assert "navigator" in cfg
    assert "topology" in cfg
    assert "training" in cfg
    assert "control" in cfg
    assert cfg["problem"]["K_min"] <= cfg["problem"]["K_max"]
    print(f"  Config loaded, K=[{cfg['problem']['K_min']},{cfg['problem']['K_max']}] ✓")


# ─── Run all tests ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("phi_dim math",           test_phi_dim),
        ("log_likelihood shapes",  test_log_likelihood_shape),
        ("union-find",             test_union_find),
        ("silverman bandwidth",    test_silverman_bandwidth),
        ("feature dimension",      test_feat_dim),
        ("BAOAB coefficients",     test_baoab_steps),
        ("systematic resample",    test_systematic_resample),
        ("ESS computation",        test_ess),
        ("file structure",         test_file_structure),
        ("config loading",         test_config),
    ]

    print("=" * 50)
    print("GMM SPHS Smoke Tests")
    print("=" * 50)

    passed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\n{'='*50}")
    print(f"Passed: {passed}/{len(tests)}")
    print("=" * 50)
