"""단위 테스트 — 방법론 함수를 손계산·완전탐색과 대조 (데이터·파이프라인 불필요)."""
from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
import pandas as pd
import pytest

from src.accessibility import band_weight, band_weights, e2sfca
from src.mvi import minmax, rank_norm, combine, compute_components
from src.optimization import solve_mclp, greedy_select, coverage_of
from src.policy_scenarios import solve_maximin, nondominated, knee_point
from src.routing import held_karp, _partition

BANDS = [(0, 1000, 1.0), (1000, 2000, 0.6), (2000, 3000, 0.25)]


# ---------- E2SFCA ----------
def test_band_weight_boundaries():
    assert band_weight(0, BANDS) == 1.0 and band_weight(999.9, BANDS) == 1.0
    assert band_weight(1000, BANDS) == 0.6 and band_weight(2999, BANDS) == 0.25
    assert band_weight(3000, BANDS) == 0.0 and band_weight(np.inf, BANDS) == 0.0
    D = np.array([[500, 1500, 2500, 3500, np.inf]])
    assert band_weights(D, BANDS).tolist() == [[1.0, 0.6, 0.25, 0.0, 0.0]]


def test_e2sfca_hand_computed():
    # 수요점 2개(인구 100, 300), 시설 1개(공급 2): 거리 500m(가중1), 1500m(가중0.6)
    D = np.array([[500.0], [1500.0]])
    A, R = e2sfca(D, np.array([100.0, 300.0]), np.array([2.0]), BANDS)
    denom = 1.0 * 100 + 0.6 * 300
    assert R[0] == pytest.approx(2 / denom)
    assert A.tolist() == pytest.approx([1.0 * 2 / denom, 0.6 * 2 / denom])


def test_e2sfca_zero_division_and_unreachable():
    D = np.array([[np.inf, 500.0], [np.inf, 5000.0]])
    A, R = e2sfca(D, np.array([10.0, 10.0]), np.array([5.0, 1.0]), BANDS)
    assert R[0] == 0.0  # 캐치먼트 내 수요가 없는 시설
    assert A[1] == 0.0 and np.isfinite(A).all()


def test_e2sfca_supply_conservation_random():
    rng = np.random.default_rng(0)
    D = rng.uniform(0, 4000, size=(20, 30))
    P = rng.uniform(100, 5000, size=20)
    S = rng.uniform(1, 10, size=30)
    A, _ = e2sfca(D, P, S, BANDS)
    reachable = (band_weights(D, BANDS) > 0).any(axis=0)
    assert (P * A).sum() == pytest.approx(S[reachable].sum())


def test_e2sfca_input_errors():
    with pytest.raises(ValueError):
        e2sfca(np.zeros((2, 2)), np.ones(3), np.ones(2))
    with pytest.raises(ValueError):
        e2sfca(np.zeros((1, 1)), np.array([-1.0]), np.ones(1))


# ---------- MVI ----------
def test_normalizations():
    assert minmax(pd.Series([3, 3, 3])).tolist() == [0.5, 0.5, 0.5]
    assert minmax(pd.Series([1, 2, 3])).tolist() == [0.0, 0.5, 1.0]
    assert rank_norm(pd.Series([10, 1, 5])).tolist() == [1.0, 0.0, 0.5]


def test_combine_forms():
    need, deficit = pd.Series([0.2, 0.8, 1.0]), pd.Series([0.5, 0.5, 0.0])
    assert combine(need, deficit, 0.5, 0.5).tolist() == pytest.approx((need * deficit).tolist())
    assert combine(need, deficit, 0.5, 0.5, "additive").tolist() == pytest.approx((0.5 * need + 0.5 * deficit).tolist())
    with pytest.raises(ValueError):
        combine(need, deficit, 0.6, 0.6)


def test_components_range_and_monotonic():
    df = pd.DataFrame({"elderly_population": [100, 200, 300, 400], "elderly_ratio": [10, 20, 15, 30],
                       "a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0], "c": [1.0, 1.0, 2.0, 2.0]})
    comp = compute_components(df, access_components={"medical": "a", "welfare": "b", "transit": "c"})
    assert ((comp >= 0) & (comp <= 1)).all().all()
    mvi = combine(comp["need"], comp["access_deficit"], 0.5, 0.5)
    df2 = df.copy()
    df2.loc[1, "elderly_population"] = 350  # 수요 증가
    comp2 = compute_components(df2, access_components={"medical": "a", "welfare": "b", "transit": "c"})
    assert combine(comp2["need"], comp2["access_deficit"], 0.5, 0.5)[1] > mvi[1]


# ---------- 최적화 (완전탐색 대조) ----------
def _toy(seed: int = 1, n_cand: int = 9, n_dem: int = 12) -> dict:
    rng = np.random.default_rng(seed)
    cover = rng.random((n_cand, n_dem)) < 0.3
    D = rng.uniform(100, 1500, size=(n_cand, n_dem))
    w = rng.uniform(1, 100, size=n_dem)
    gu = np.array(["수정구", "중원구", "분당구"] * (n_dem // 3))
    return {"cover": cover, "D": D, "vuln_w": w, "elderly": w, "radius_m": 1500.0,
            "dong_codes": np.array([f"{i:010d}" for i in range(n_dem)]), "gu": gu}


@pytest.mark.parametrize("k", [1, 2, 3])
def test_mclp_matches_brute_force(k):
    data = _toy()
    best = max(data["vuln_w"][coverage_of(list(s), data)].sum() for s in combinations(range(9), k))
    r = solve_mclp(data, k)
    assert r["optimal"]
    assert data["vuln_w"][r["covered"]].sum() == pytest.approx(best)
    assert data["vuln_w"][coverage_of(greedy_select(data, k), data)].sum() <= best + 1e-9


def test_mclp_tiebreak_deterministic():
    data = _toy(seed=3)
    assert solve_mclp(data, 2)["selected_idx"] == solve_mclp(data, 2)["selected_idx"]


def test_maximin_matches_brute_force():
    data = _toy(seed=2)

    def min_gu(sel):
        cov = coverage_of(list(sel), data)
        return min(data["vuln_w"][(data["gu"] == g) & cov].sum() / data["vuln_w"][data["gu"] == g].sum()
                   for g in set(data["gu"]))
    best = max(min_gu(s) for s in combinations(range(9), 3))
    r = solve_maximin(data, 3)
    assert r["optimal"] and min_gu(r["selected_idx"]) == pytest.approx(best)


def test_pareto_helpers():
    pts = [(10, 0.3), (20, 0.5), (15, 0.2), (30, 0.5)]
    assert nondominated(pts) == [True, True, False, False]
    assert knee_point(np.array([1, 2, 3, 4, 5.0]), np.array([0.1, 0.6, 0.75, 0.8, 0.82])) == 1


# ---------- 노선 (완전탐색 대조) ----------
def test_held_karp_matches_permutations():
    rng = np.random.default_rng(5)
    pts = rng.uniform(0, 10, size=(6, 2))
    D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(2))
    cost, _ = held_karp(D)
    full = (1 << 5) - 1
    brute = min(D[0, p[0] + 1] + sum(D[a + 1, b + 1] for a, b in zip(p[:-1], p[1:])) + D[p[-1] + 1, 0]
                for p in permutations(range(5)))
    assert cost[full] == pytest.approx(brute)


def test_partition_respects_feasibility():
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 10, size=(5, 2))
    D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(2))
    cost, _ = held_karp(D)
    n = 4
    feas = np.array([bin(m).count("1") <= 2 for m in range(1 << n)])
    feas[0] = False
    total, parts = _partition(cost, feas, n, 2)
    assert len(parts) == 2 and all(bin(p).count("1") <= 2 for p in parts)
    assert sum(cost[p] for p in parts) == pytest.approx(total)
    assert not np.isfinite(_partition(cost, feas, n, 1)[0])  # 1대로는 불가능


# ---------- 정책 C 완전탐색 (독립 브루트포스 대조) ----------
def _toy_policy_c(seed: int = 11, per_gu: int = 3, n_dem: int = 12):
    rng = np.random.default_rng(seed)
    w = rng.uniform(1, 100, size=n_dem)
    bits = np.uint64(1) << np.arange(n_dem, dtype=np.uint64)
    red = {"per_gu": {}, "subset_km": {}, "subset_mask": {}, "subset_n": {}}
    cover, j = {}, 0
    for g in ["분당구", "수정구", "중원구"]:
        pts = rng.uniform(0, 10, size=(per_gu + 1, 2))
        cost, _ = held_karp(np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(2)))
        rows = rng.random((per_gu, n_dem)) < 0.35
        masks = np.array([np.bitwise_or.reduce(np.where(
            np.array([(m >> k) & 1 for k in range(per_gu)], bool)[:, None] & rows, bits[None, :], np.uint64(0)).ravel())
            for m in range(1 << per_gu)], dtype=np.uint64)
        red["per_gu"][g] = list(range(j, j + per_gu))
        red["subset_km"][g], red["subset_mask"][g] = cost, masks
        red["subset_n"][g] = np.array([bin(m).count("1") for m in range(1 << per_gu)])
        for k in range(per_gu):
            cover[j + k] = rows[k]
        j += per_gu
    return {"vuln_w": w}, red, cover


def _brute_policy_c(data, red, cover):
    from itertools import product as iproduct
    gus = sorted(red["per_gu"])
    out = []
    for subs in iproduct(*[range(len(red["subset_km"][g])) for g in gus]):
        sel = [red["per_gu"][g][k] for g, m in zip(gus, subs) for k in range(len(red["per_gu"][g])) if (m >> k) & 1]
        km = sum(red["subset_km"][g][m] for g, m in zip(gus, subs))
        cov = np.zeros(len(data["vuln_w"]), bool)
        for s in sel:
            cov |= cover[s]
        out.append((sel, km, float(data["vuln_w"][cov].sum())))
    return out


@pytest.mark.parametrize("seed", [11, 12])
def test_policy_c_enumeration_matches_brute_force(seed):
    from src.policy_scenarios import min_km_for_coverage, orienteering_frontier
    data, red, cover = _toy_policy_c(seed)
    allc = _brute_policy_c(data, red, cover)
    W = data["vuln_w"].sum()
    for target in [0.3, 0.6, 0.9]:
        feas = [(km, c) for sel, km, c in allc if 1 <= len(sel) <= 10 and c >= target * W - 1e-9]
        r = min_km_for_coverage(data, red, target)
        if not feas:
            assert r["status"] == "Infeasible"
            continue
        assert r["route_km"] == pytest.approx(min(km for km, _ in feas), abs=1e-3)
    budgets = [5.0, 12.0, 25.0, 60.0]
    for L, r in zip(budgets, orienteering_frontier(data, red, budgets)):
        feas = [c for sel, km, c in allc if 1 <= len(sel) <= 10 and km <= L + 1e-9]
        if feas:
            assert r["objective"] == pytest.approx(max(feas), abs=1e-3) and r["route_km"] <= L + 1e-6
