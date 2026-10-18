"""정책 대안 A/B/C — 목적함수·제약·결정변수가 서로 다른 3개 최적화 모형 + 동일 KPI 비교.

A 공공효과 우선 : MCLP  max Σ vuln_w·y          s.t. Σx = K_MAIN
B 형평성 우선   : Maximin max t (+1e-4·전체커버)  s.t. Σx = K_MAIN, t ≤ 구별 취약커버율 (3개 구)
C 운영효율 우선 : 대표안 = 정책 A와 같은 취약커버율 이상을 최소 vehicle-km로 (같은 효과, 최소 운행)
                  min Σ_구 순환노선 거리  s.t. 취약커버율 ≥ A, 1 ≤ Σx ≤ POLICY_C_MAX_STOPS
                  Pareto 곡선 = 커버리지 Orienteering: max Σ vuln_w·y s.t. Σ_구 순환노선 거리 ≤ 예산 L(km)
                  노선거리는 선택 정류장 집합의 구별 최단 순환거리(Held-Karp 정확 TSP) — 입지와 노선을 함께 결정.
                  풀이: 구별 후보 축약(최적해 등장 정류장 빈도순 → 동일 커버집합 대표점 → 커버·거점왕복거리 지배
                  제거 → 커버 수요 상위, 구별 POLICY_C_CANDIDATES_PER_GU개) 후, 3개 구 부분집합 조합 전체(2^8)^3를
                  완전탐색 → 축약 후보 내 정확해(전체 후보 전역최적은 주장하지 않음).
                  축약은 구별 8개 상한 때문에 정책 A의 해를 항상 포함하지는 않는다. 대신 A의 해(목표 커버를 자명하게
                  만족하는 실행가능해)를 같은 Held-Karp 노선거리로 평가해 비교 후보에 넣으므로 C ≤ A(vehicle-km)가 보장된다.
                  예산 L을 바꿔가며 (vehicle-km, 커버리지) Pareto 곡선을 만들고 비지배 여부를 검증,
                  곡선의 knee(max-distance-to-chord)를 대표안으로 제시한다.
A/B/C 모두 선정 정류장을 src/routing.py의 같은 노선 모형으로 평가해 vehicle-km·순환시간을 계산한다.
동점 해소: A/B는 src/optimization.py와 같은 2단계(최적값 고정 후 커버거리 최소), C는 최적 커버리지
고정 후 노선거리 최소 → 모든 대안이 결정적.

출력: outputs/tables/policy_comparison.csv, outputs/json/policy_results.json,
      outputs/tables/policy_C_pareto.csv, outputs/json/policy_C_pareto.json,
      outputs/tables/policy_{A,B,C}_selected.csv,
      outputs/tables/before_after_by_dong.csv, outputs/tables/before_after_by_gu.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pulp

from src import config as C
from src import network as N
from src.optimization import load_cover_data, solve_mclp, coverage_of, _two_stage
from src.routing import RouteParams, plan_routes, load_depots, dong_gu_map, held_karp
from src.kpi import coverage_kpis, route_kpis, reference_sets

POLICY_C_CANDIDATES_PER_GU = 8  # 구별 2^8 부분집합 → 3개 구 조합 1,677만 개 완전탐색
POLICY_META = {
    "A": {"name": "A_공공효과우선", "goal": "제한된 거점으로 취약 고령수요 커버 최대",
          "objective": "max Σ_i vuln_w_i·y_i", "constraints": f"Σx = {C.K_MAIN}; y_i ≤ Σ_(j∈N(i)) x_j",
          "decision_variables": "x_j(정류장 선정), y_i(동 커버)"},
    "B": {"name": "B_형평성우선", "goal": "3개 구 중 가장 낮은 구의 취약커버율을 최대 (지역 균형)",
          "objective": "max t + 1e-4·(전체 취약커버율)",
          "constraints": f"Σx = {C.K_MAIN}; t ≤ 구별 취약커버율(수정·중원·분당)",
          "decision_variables": "x_j, y_i, t(최소 구 커버율)"},
    "C": {"name": "C_운영효율우선", "goal": "정책 A와 같은 공공효과를 가장 짧은 차량 운행거리로",
          "objective": "min Σ_구 최단 순환노선 거리 (vehicle-km)",
          "constraints": f"취약커버율 ≥ 정책 A 값; 1 ≤ Σx ≤ {C.POLICY_C_MAX_STOPS} (예산별 Pareto: 노선거리 ≤ L에서 커버 최대)",
          "decision_variables": "x_j(정류장 선정)와 구 보건소별 순환노선(방문 순서) — 노선거리가 제약에 직접 들어감"},
}


def solve_maximin(data: dict, k: int) -> dict:
    Ncand, Ndong = data["cover"].shape
    gu = data["gu"]
    zones = sorted(set(gu.tolist()))
    Wz = {z: float(data["vuln_w"][gu == z].sum()) for z in zones}
    prob = pulp.LpProblem("MaximinEquity", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(Ncand)]
    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(Ndong)]
    t = pulp.LpVariable("t_min_cov", lowBound=0, upBound=1)
    total = pulp.lpSum(float(data["vuln_w"][i]) * y[i] for i in range(Ndong)) / float(data["vuln_w"].sum())
    obj = t + 1e-4 * total
    prob += obj
    prob += pulp.lpSum(x) == k
    for i in range(Ndong):
        cov_i = np.flatnonzero(data["cover"][:, i])
        prob += (y[i] <= pulp.lpSum(x[j] for j in cov_i)) if len(cov_i) else (y[i] == 0)
    for z in zones:
        idx = np.flatnonzero(gu == z)
        prob += t <= pulp.lpSum(float(data["vuln_w"][i]) * y[i] for i in idx) / Wz[z]
    status, opt, sel = _two_stage(prob, obj, x, data, f"Maximin_k{k}")
    return {"status": status, "optimal": status == "Optimal", "selected_idx": sel, "objective": opt}


# ---------------- Policy C: coverage orienteering ----------------
def _cand_gu(data: dict) -> np.ndarray:
    gmap = dong_gu_map()
    codes = data["cand"]["official_dong_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    return codes.map(gmap).to_numpy()


def seed_candidates(data: dict) -> list[int]:
    """후보 축약 시 우선 포함할 정류장: MCLP 최적해(k=1..10)와 정책 B(형평) 최적해에 등장한 정류장.

    등장 빈도가 높은 순. 구별 상한(POLICY_C_CANDIDATES_PER_GU)에서 잘리므로 특정 해(예: 정책 A)의
    정류장이 모두 축약 후보에 남는다는 보장은 없다 — run_policies가 A의 해를 별도 비교 후보로 평가한다.
    """
    from collections import Counter
    cnt: Counter = Counter()
    for k in C.COVERAGE_CURVE_K:
        cnt.update(solve_mclp(data, k)["selected_idx"])
    cnt.update(solve_maximin(data, C.K_MAIN)["selected_idx"])
    return [j for j, _ in sorted(cnt.items(), key=lambda t: (-t[1], t[0]))]


def reduce_candidates(data: dict, per_gu: int = POLICY_C_CANDIDATES_PER_GU, seeds: list[int] | None = None) -> dict:
    """구별 후보 축약 + 부분집합별 정확 노선거리·커버 비트마스크 사전계산.

    축약: ① seed(최적해 등장 정류장)를 빈도순으로 우선 포함 → ② 나머지는 커버가 있는 정류장 중
          동일 커버집합의 거점 왕복거리 최소 대표 → (커버 ⊇, 왕복거리 ≤) 지배 제거 → 커버 수요 큰 순으로 채움.
          구별 per_gu개.
    사전계산: Held-Karp 1회로 모든 부분집합의 최단 순환거리(km), 부분집합의 동 커버 비트마스크(uint64).
    """
    dep = load_depots().set_index("gu")
    drive = N.load("drive")
    cgu = _cand_gu(data)
    cand = data["cand"]
    cov_w = data["cover"].astype(float) @ data["vuln_w"]
    bits = (np.uint64(1) << np.arange(data["cover"].shape[1], dtype=np.uint64))
    out = {"depots": dep, "per_gu": {}, "D": {}, "subset_km": {}, "subset_mask": {}, "subset_n": {}}
    for g in sorted(dep.index):
        idx = np.flatnonzero((cgu == g) & (cov_w > 0))
        d = dep.loc[g]
        lon = np.r_[d["lon"], cand.loc[idx, "경도"].to_numpy(float)]
        lat = np.r_[d["lat"], cand.loc[idx, "위도"].to_numpy(float)]
        rt = drive.distance_matrix(lon[:1], lat[:1], lon[1:], lat[1:])[0] + \
            drive.distance_matrix(lon[1:], lat[1:], lon[:1], lat[:1])[:, 0]
        groups: dict[bytes, int] = {}
        for pos, j in enumerate(idx):
            key = data["cover"][j].tobytes()
            if key not in groups or rt[pos] < rt[groups[key]]:
                groups[key] = pos
        reps = list(groups.values())
        keep = []
        for a in reps:
            ca = data["cover"][idx[a]]
            dominated = any(b != a and (data["cover"][idx[b]] >= ca).all() and rt[b] <= rt[a]
                            and ((data["cover"][idx[b]] > ca).any() or rt[b] < rt[a]) for b in reps)
            if not dominated:
                keep.append(a)
        keep = sorted(keep, key=lambda a: (-cov_w[idx[a]], rt[a]))
        pos_of = {int(j): p for p, j in enumerate(idx)}
        seeded = [pos_of[j] for j in (seeds or []) if j in pos_of]
        keep = (seeded + [a for a in keep if a not in seeded])[:per_gu]
        sel = idx[keep]
        lon2 = np.r_[d["lon"], cand.loc[sel, "경도"].to_numpy(float)]
        lat2 = np.r_[d["lat"], cand.loc[sel, "위도"].to_numpy(float)]
        Dkm = drive.distance_matrix(lon2, lat2, lon2, lat2) / 1000.0
        cost, _ = held_karp(Dkm)
        n = len(sel)
        cmask = np.array([np.bitwise_or.reduce(bits[data["cover"][j]]) if data["cover"][j].any() else np.uint64(0)
                          for j in sel], dtype=np.uint64)
        masks = np.zeros(1 << n, dtype=np.uint64)
        counts = np.zeros(1 << n, dtype=np.int64)
        for m in range(1, 1 << n):
            low = (m & -m).bit_length() - 1
            masks[m] = masks[m & (m - 1)] | cmask[low]
            counts[m] = counts[m & (m - 1)] + 1
        out["per_gu"][g], out["D"][g] = sel.tolist(), Dkm
        out["subset_km"][g], out["subset_mask"][g], out["subset_n"][g] = cost, masks, counts
    return out


def _weighted_cover(masks: np.ndarray, w: np.ndarray) -> np.ndarray:
    """uint64 커버 비트마스크 배열 → 취약가중 커버 합 (바이트 단위 조회표)."""
    total = np.zeros(masks.shape, dtype=float)
    for b in range((len(w) + 7) // 8):
        lut = np.array([sum(w[8 * b + k] for k in range(8) if (v >> k) & 1 and 8 * b + k < len(w))
                        for v in range(256)])
        total += lut[((masks >> np.uint64(8 * b)) & np.uint64(255)).astype(np.int64)]
    return total


def orienteering_frontier(data: dict, red: dict, budgets_km: list[float],
                          max_stops: int = C.POLICY_C_MAX_STOPS) -> list[dict]:
    """축약 후보의 모든 구별 부분집합 조합을 완전탐색해 예산별 정확해를 구한다.

    예산 L마다: Σ구 노선거리 ≤ L, 정류장 수 1~max_stops 중 취약가중 커버 최대, 동률이면 노선거리 최소,
    그래도 동률이면 열거 순서상 먼저 나온 조합(결정적).
    """
    gus = sorted(red["per_gu"])
    if len(gus) != 3:
        raise ValueError("정책 C 완전탐색은 3개 구를 가정")
    w = data["vuln_w"]
    g1, g2, g3 = gus
    km12 = (red["subset_km"][g1][:, None] + red["subset_km"][g2][None, :]).ravel()
    n12 = (red["subset_n"][g1][:, None] + red["subset_n"][g2][None, :]).ravel()
    m12 = (red["subset_mask"][g1][:, None] | red["subset_mask"][g2][None, :]).ravel()
    n2 = len(red["subset_km"][g2])
    best = [{"cov": -1.0, "km": np.inf, "p": -1, "s3": -1} for _ in budgets_km]
    for s3 in range(len(red["subset_km"][g3])):
        km = km12 + red["subset_km"][g3][s3]
        n = n12 + red["subset_n"][g3][s3]
        cov = _weighted_cover(m12 | red["subset_mask"][g3][s3], w)
        ok_n = (n >= 1) & (n <= max_stops)
        for bi, L in enumerate(budgets_km):
            valid = ok_n & (km <= L + 1e-9)
            if not valid.any():
                continue
            cv = np.where(valid, cov, -1.0)
            top = cv.max()
            cand_idx = np.flatnonzero(cv == top)
            p = int(cand_idx[np.argmin(km[cand_idx])])
            b = best[bi]
            if top > b["cov"] or (top == b["cov"] and km[p] < b["km"] - 1e-12):
                best[bi] = {"cov": float(top), "km": float(km[p]), "p": p, "s3": s3}
    res = []
    for L, b in zip(budgets_km, best):
        if b["p"] < 0:
            res.append({"budget_km": L, "status": "Infeasible", "optimal": True, "selected_idx": [], "route_km": None})
            continue
        subs = {g1: b["p"] // n2, g2: b["p"] % n2, g3: b["s3"]}
        sel = sorted(red["per_gu"][g][k] for g, m in subs.items() for k in range(len(red["per_gu"][g])) if (m >> k) & 1)
        res.append({"budget_km": L, "status": "Optimal", "optimal": True, "selected_idx": sel,
                    "route_km": round(b["km"], 3), "objective": round(b["cov"], 3)})
    return res


def solve_orienteering(data: dict, red: dict, budget_km: float, max_stops: int = C.POLICY_C_MAX_STOPS) -> dict:
    return orienteering_frontier(data, red, [budget_km], max_stops)[0]


def min_km_for_coverage(data: dict, red: dict, target_cov: float, max_stops: int = C.POLICY_C_MAX_STOPS) -> dict:
    """취약커버율 ≥ target_cov를 만족하는 조합 중 구별 순환노선 거리 합 최소 (동률이면 커버 큰 쪽) — 완전탐색."""
    gus = sorted(red["per_gu"])
    g1, g2, g3 = gus
    w = data["vuln_w"]
    need = target_cov * float(w.sum()) - 1e-9 * float(w.sum())
    km12 = (red["subset_km"][g1][:, None] + red["subset_km"][g2][None, :]).ravel()
    n12 = (red["subset_n"][g1][:, None] + red["subset_n"][g2][None, :]).ravel()
    m12 = (red["subset_mask"][g1][:, None] | red["subset_mask"][g2][None, :]).ravel()
    n2 = len(red["subset_km"][g2])
    best = {"km": np.inf, "cov": -1.0, "p": -1, "s3": -1}
    for s3 in range(len(red["subset_km"][g3])):
        km = km12 + red["subset_km"][g3][s3]
        n = n12 + red["subset_n"][g3][s3]
        cov = _weighted_cover(m12 | red["subset_mask"][g3][s3], w)
        valid = (n >= 1) & (n <= max_stops) & (cov >= need)
        if not valid.any():
            continue
        kv = np.where(valid, km, np.inf)
        low = kv.min()
        cand_idx = np.flatnonzero(kv == low)
        p_ = int(cand_idx[np.argmax(cov[cand_idx])])
        if low < best["km"] - 1e-12 or (low == best["km"] and cov[p_] > best["cov"]):
            best = {"km": float(low), "cov": float(cov[p_]), "p": p_, "s3": s3}
    if best["p"] < 0:
        return {"status": "Infeasible", "optimal": True, "selected_idx": [], "route_km": None, "target": target_cov}
    subs = {g1: best["p"] // n2, g2: best["p"] % n2, g3: best["s3"]}
    sel = sorted(red["per_gu"][g][k] for g, m in subs.items() for k in range(len(red["per_gu"][g])) if (m >> k) & 1)
    return {"status": "Optimal", "optimal": True, "selected_idx": sel, "route_km": round(best["km"], 3),
            "objective": round(best["cov"], 3), "target": target_cov}


def exact_route_km(data: dict, selected: list[int]) -> float:
    """선택 정류장 집합의 구별 최단 순환거리 합(km) — reduce_candidates와 같은 Held-Karp·차량망 거리 정의."""
    dep = load_depots().set_index("gu")
    drive = N.load("drive")
    cgu = _cand_gu(data)
    cand = data["cand"]
    total = 0.0
    for g in sorted(set(cgu[selected])):
        idx = [j for j in selected if cgu[j] == g]
        lon = np.r_[dep.loc[g, "lon"], cand.loc[idx, "경도"].to_numpy(float)]
        lat = np.r_[dep.loc[g, "lat"], cand.loc[idx, "위도"].to_numpy(float)]
        cost, _ = held_karp(drive.distance_matrix(lon, lat, lon, lat) / 1000.0)
        total += float(cost[-1])
    return total


def nondominated(points: list[tuple[float, float]]) -> list[bool]:
    """(비용↓, 효과↑) 기준 비지배 여부."""
    res = []
    for i, (c, e) in enumerate(points):
        dom = any((c2 <= c and e2 >= e) and (c2 < c or e2 > e) for j, (c2, e2) in enumerate(points) if j != i)
        res.append(not dom)
    return res


def knee_point(cost: np.ndarray, eff: np.ndarray) -> int:
    """정규화 곡선에서 양 끝점을 잇는 현(chord)과 가장 먼 점의 인덱스 (max-distance-to-chord)."""
    if len(cost) < 3:
        return int(np.argmax(eff))
    x = (cost - cost.min()) / (np.ptp(cost) or 1)
    y = (eff - eff.min()) / (np.ptp(eff) or 1)
    x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
    d = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / (np.hypot(y2 - y1, x2 - x1) or 1)
    return int(np.argmax(d))


# ---------------- 평가·비교 ----------------
def _service_walk_m(data: dict, selected: list[int]) -> np.ndarray:
    """동 수요 대표점 → 가장 가까운 신규 서비스 정류장 보행 네트워크 거리(m)."""
    if not selected:
        return np.full(len(data["dong_codes"]), np.inf)
    import geopandas as gpd
    dong = gpd.read_file(C.DONG_MASTER, encoding="utf-8")
    lon, lat = N.dong_centroids_lonlat(dong)
    st = data["cand"].iloc[selected]
    D = N.load("walk").distance_matrix(lon, lat, st["경도"].to_numpy(float), st["위도"].to_numpy(float))
    return D.min(axis=1)


def evaluate_policy(key: str, selected: list[int], data: dict, refs: dict, params: RouteParams,
                    with_paths: bool = False) -> dict:
    cov = coverage_kpis(selected, data, refs)
    stops = data["cand"].iloc[selected]
    plan = plan_routes(stops, params, with_paths=with_paths).as_dict()
    walk = _service_walk_m(data, selected)
    covered = coverage_of(selected, data)
    w = data["vuln_w"]
    kp = {**cov, **route_kpis(cov, plan),
          "mean_walk_to_service_m_covered": round(float((walk[covered] * w[covered]).sum() / w[covered].sum()), 1)
          if covered.any() else None}
    return {"key": key, **POLICY_META[key], "kpis": kp, "routes": plan["routes"],
            "vehicles_required_by_gu": plan["vehicles_required_by_gu"],
            "selected_stops": [{"stop_id": str(r["정류장번호(ID)"]), "name": r["정류장명"],
                                "lon": float(r["경도"]), "lat": float(r["위도"]),
                                "dong": r["official_dong_name"]} for _, r in stops.iterrows()],
            "covered_dongs": sorted(data["dong_codes"][covered].tolist())}


def before_after(data: dict, policies: dict[str, list[int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """동·구별 before(전담 서비스 없음) → after(정책별) — 지표별 baseline/after/차이/증감률."""
    rows = []
    w, e = data["vuln_w"], data["elderly"]
    for key, sel in policies.items():
        cov = coverage_of(sel, data)
        walk = _service_walk_m(data, sel)
        for i, code in enumerate(data["dong_codes"]):
            base = {"dong_code": code, "dong_name": data["dong_names"][i], "gu": data["gu"][i], "policy": key}
            rows.append({**base, "metric": "served_elderly", "baseline": 0.0,
                         "after": float(e[i] * cov[i])})
            rows.append({**base, "metric": "uncovered_vulnerable_demand", "baseline": round(float(w[i]), 2),
                         "after": round(float(w[i] * (1 - cov[i])), 2)})
            rows.append({**base, "metric": "walk_to_service_m", "baseline": np.nan,
                         "after": round(float(walk[i]), 1) if np.isfinite(walk[i]) else np.nan})
    df = pd.DataFrame(rows)
    df["difference"] = df["after"] - df["baseline"]
    df["percent_change"] = np.where(df["baseline"] > 0, (df["difference"] / df["baseline"] * 100).round(2), np.nan)
    gu = (df[df["metric"] != "walk_to_service_m"].groupby(["gu", "policy", "metric"], as_index=False)
          [["baseline", "after"]].sum())
    gu["difference"] = gu["after"] - gu["baseline"]
    gu["percent_change"] = np.where(gu["baseline"] > 0, (gu["difference"] / gu["baseline"] * 100).round(2), np.nan)
    return df, gu


def run_policies(k_main: int = C.K_MAIN, radius_m: float = C.COVERAGE_RADIUS_M) -> dict:
    data = load_cover_data(radius_m)
    refs = reference_sets(data)
    params = RouteParams()

    a = solve_mclp(data, k_main)
    b = solve_maximin(data, k_main)
    for name, r in [("A", a), ("B", b)]:
        if not r["optimal"]:
            raise RuntimeError(f"Policy {name} solver 비최적({r['status']})")

    red = reduce_candidates(data, seeds=seed_candidates(data))
    pareto = []
    for r in orienteering_frontier(data, red, C.POLICY_C_BUDGETS_KM):
        L = r["budget_km"]
        cov = coverage_kpis(r["selected_idx"], data, refs) if r["selected_idx"] else None
        pareto.append({"budget_km": L, "status": r["status"], "optimal": r["optimal"],
                       "n_stops": len(r["selected_idx"]), "route_km": r["route_km"],
                       "vuln_coverage": cov["vuln_coverage"] if cov else 0.0,
                       "elderly_coverage": cov["elderly_coverage"] if cov else 0.0,
                       "min_gu_coverage": cov["min_gu_coverage"] if cov else 0.0,
                       "selected_idx": r["selected_idx"]})
        print(f"Policy C budget {L}km: status={r['status']} optimal={r['optimal']} "
              f"km={r['route_km']} stops={len(r['selected_idx'])} cov={pareto[-1]['vuln_coverage']}")
    pdf = pd.DataFrame(pareto)
    valid = pdf[pdf["route_km"].notna() & (pdf["n_stops"] > 0)].copy()
    uniq = valid.drop_duplicates(subset=["route_km", "vuln_coverage"])
    nd = nondominated(list(zip(uniq["route_km"], uniq["vuln_coverage"])))
    uniq = uniq.assign(nondominated=nd)
    # 예산이 달라도 같은 해(노선거리·커버 동일)면 같은 점 — 대표 점의 비지배 여부를 공유, 포화 여부 표시
    key = dict(zip(zip(uniq["route_km"], uniq["vuln_coverage"]), uniq["nondominated"]))
    first = dict(zip(zip(uniq["route_km"], uniq["vuln_coverage"]), uniq["budget_km"]))
    pdf["nondominated"] = [bool(key.get((k, c), False)) for k, c in zip(pdf["route_km"], pdf["vuln_coverage"])]
    pdf["same_solution_as_budget"] = [first.get((k, c)) if first.get((k, c)) != b else None
                                      for k, c, b in zip(pdf["route_km"], pdf["vuln_coverage"], pdf["budget_km"])]
    front = uniq[uniq["nondominated"]].sort_values("route_km")
    ki = knee_point(front["route_km"].to_numpy(float), front["vuln_coverage"].to_numpy(float))
    knee_row = front.iloc[ki]
    pdf["is_knee"] = pdf["budget_km"] == knee_row["budget_km"]
    # 정책 C 대표안: 정책 A와 같은 공공효과(취약커버율) 이상을 최소 vehicle-km로
    target = coverage_kpis(a["selected_idx"], data, refs)["vuln_coverage"]
    c_res = min_km_for_coverage(data, red, target)
    # 축약 후보가 A의 해를 포함하지 않을 수 있으므로, A의 해(목표 커버를 자명하게 만족)를 같은 노선거리 정의로
    # 평가해 비교한다 → 대표안은 항상 A 이하의 vehicle-km (A가 더 짧으면 A의 해를 대표안으로 채택하고 기록)
    reduced = {j for v in red["per_gu"].values() for j in v}
    a_in_reduced = set(a["selected_idx"]) <= reduced
    a_km = round(exact_route_km(data, a["selected_idx"]), 3)
    use_a = len(a["selected_idx"]) <= C.POLICY_C_MAX_STOPS and (
        not c_res["selected_idx"] or a_km < c_res["route_km"] - 1e-9)
    if use_a:
        c_res = {"status": "Optimal", "optimal": True, "selected_idx": sorted(a["selected_idx"]), "route_km": a_km,
                 "target": target}
    if not c_res["selected_idx"]:
        raise RuntimeError(f"Policy C: 목표 취약커버율 {target} 달성 불가 (축약 후보·최대 정류장 수 확인)")
    c_sel = c_res["selected_idx"]

    res = {"A": evaluate_policy("A", a["selected_idx"], data, refs, params, with_paths=True),
           "B": evaluate_policy("B", b["selected_idx"], data, refs, params, with_paths=True),
           "C": evaluate_policy("C", c_sel, data, refs, params, with_paths=True)}
    res["A"]["solver"] = {"status": a["status"], "objective": a["objective"]}
    res["B"]["solver"] = {"status": b["status"], "objective": b["objective"]}
    res["C"]["solver"] = {"status": c_res["status"], "optimal": bool(c_res["optimal"]),
                          "target_vuln_coverage": target, "enumerated_route_km": float(c_res["route_km"]),
                          "A_solution_in_reduced_candidates": bool(a_in_reduced),
                          "A_solution_route_km_exact": a_km, "representative_is_A_solution": bool(use_a),
                          "note": f"축약 후보(구별 {POLICY_C_CANDIDATES_PER_GU}개) 조합 완전탐색 정확해 + 정책 A 해와 비교"
                                  " (전체 후보 전역최적 아님, 구별 단일 순환 노선거리 기준 C ≤ A 보장)"}

    # 동일 비용 비교: A의 노선거리를 예산으로 C를 풀었을 때
    same = solve_orienteering(data, red, res["A"]["kpis"]["total_vehicle_km"])
    same_cov = coverage_kpis(same["selected_idx"], data, refs) if same["selected_idx"] else None
    sets = {k: set(v["covered_dongs"]) for k, v in res.items()}
    sel_sets = {k: {s["stop_id"] for s in v["selected_stops"]} for k, v in res.items()}
    uniqueness = {f"{p}_vs_{q}": {"same_selected_stops": sel_sets[p] == sel_sets[q],
                                  "jaccard_stops": round(len(sel_sets[p] & sel_sets[q]) / len(sel_sets[p] | sel_sets[q]), 3),
                                  "jaccard_covered_dongs": round(len(sets[p] & sets[q]) / len(sets[p] | sets[q]), 3)}
                  for p, q in [("A", "B"), ("A", "C"), ("B", "C")]}

    kpi_cols = ["n_new_stops", "vuln_coverage", "elderly_coverage", f"top{C.TOP_K_VULNERABLE}_coverage",
                "hotspot_coverage", "min_gu_coverage", "gu_coverage_gap", "covered_elderly",
                "total_vehicle_km", "max_cycle_min", "vehicles_used", "covered_elderly_per_km",
                "vuln_coverage_per_10km", "mean_walk_to_service_m_covered"]
    comp = pd.DataFrame([{"policy": v["name"], "goal": v["goal"], "objective": v["objective"],
                          "constraints": v["constraints"], "decision_variables": v["decision_variables"],
                          **{c: v["kpis"][c] for c in kpi_cols},
                          "gu_coverages": json.dumps(v["kpis"]["gu_coverages"], ensure_ascii=False)}
                         for v in res.values()])
    comp.to_csv(C.TABLES / "policy_comparison.csv", index=False, encoding="utf-8-sig")
    for k, sel in [("A", a["selected_idx"]), ("B", b["selected_idx"]), ("C", c_sel)]:
        data["cand"].iloc[sel].to_csv(C.TABLES / f"policy_{k}_selected.csv", index=False, encoding="utf-8-sig")

    pdf_out = pdf.drop(columns=["selected_idx"])
    pdf_out.to_csv(C.TABLES / "policy_C_pareto.csv", index=False, encoding="utf-8-sig")
    pareto_json = {"budgets_km": C.POLICY_C_BUDGETS_KM, "knee_budget_km": float(knee_row["budget_km"]),
                   "representative": {"definition": "정책 A 취약커버율 이상을 만족하는 최소 vehicle-km 조합",
                                      "target_vuln_coverage": target, "route_km": c_res["route_km"]},
                   "method": "예산별 커버리지 Orienteering — 축약 후보 부분집합 완전탐색 + Held-Karp → 비지배 점 → knee",
                   "all_nondominated": bool(pdf_out.loc[pdf_out["route_km"].notna(), "nondominated"].all()),
                   "candidate_reduction": {g: len(v) for g, v in red["per_gu"].items()},
                   "rows": json.loads(pdf_out.to_json(orient="records", force_ascii=False))}
    (C.JSON_DIR / "policy_C_pareto.json").write_text(json.dumps(pareto_json, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")

    ba, ba_gu = before_after(data, {"A": a["selected_idx"], "B": b["selected_idx"], "C": c_sel})
    ba.to_csv(C.TABLES / "before_after_by_dong.csv", index=False, encoding="utf-8-sig")
    ba_gu.to_csv(C.TABLES / "before_after_by_gu.csv", index=False, encoding="utf-8-sig")

    ka, kb, kc = res["A"]["kpis"], res["B"]["kpis"], res["C"]["kpis"]
    tradeoff = (f"A(공공효과): 정류장 {ka['n_new_stops']}개로 취약수요 커버 {ka['vuln_coverage']:.3f}, "
                f"운행 {ka['total_vehicle_km']:.1f}km. "
                f"B(형평): 최소 구 커버 {kb['min_gu_coverage']:.3f}(A는 {ka['min_gu_coverage']:.3f}), "
                f"취약수요 커버 {kb['vuln_coverage']:.3f}. "
                f"C(운영효율): A 이상의 취약수요 커버 {kc['vuln_coverage']:.3f}를 운행 {kc['total_vehicle_km']:.1f}km"
                f"(A 대비 {kc['total_vehicle_km'] - ka['total_vehicle_km']:+.1f}km)로 달성, 대신 정류장 {kc['n_new_stops']}개. "
                "우열 판단은 의사결정자의 몫.")
    out = {"policies": res, "tradeoff": tradeoff, "uniqueness_check": uniqueness,
           "same_cost_comparison": {"budget_km": res["A"]["kpis"]["total_vehicle_km"],
                                    "C_status": same["status"], "C_optimal": same["optimal"],
                                    "C_vuln_coverage": same_cov["vuln_coverage"] if same_cov else None,
                                    "A_vuln_coverage": ka["vuln_coverage"],
                                    "C_n_stops": len(same["selected_idx"])},
           "route_assumptions": {"speed_kmh": params.speed_kmh, "dwell_min_per_stop": params.dwell_min,
                                 "vehicles_per_depot_min": params.vehicles_per_depot,
                                 "max_cycle_min": params.max_cycle_min,
                                 "depots": load_depots()[["gu", "depot_name", "coord_source"]].to_dict("records")},
           "method_version": C.METHOD_VERSION}
    (C.JSON_DIR / "policy_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(tradeoff)
    return out


if __name__ == "__main__":
    run_policies()
