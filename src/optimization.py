"""MCLP — 제한된 신규 이동지원 거점 k개로 취약 고령수요 커버리지 최대화.

수요 i   : 행정동 수요 대표점 50개, 가중치 vuln_w_i = 65세 이상 인구_i × MVI_i (primary 시나리오)
후보 j   : 실존 버스정류장 중 좌표·행정동 매칭 검증 통과분 (임의 좌표 생성 없음)
커버     : 후보 j – 동 수요 대표점 i(동 내 정류장 medoid, src/network.demand_points) 직선거리 ≤ COVERAGE_RADIUS_M
모형     : max Σ_i vuln_w_i·y_i   s.t. Σ_j x_j = k,  y_i ≤ Σ_{j∈N(i)} x_j,  x,y ∈ {0,1}
동점 처리: 같은 최적 커버리지를 내는 후보 조합이 여러 개일 수 있으므로(수요점 50개 기준),
           2단계로 1단계 최적값을 제약으로 고정하고 '선정 정류장과 그 정류장이 커버하는 수요 사이의
           취약가중 평균거리' 합을 최소화한다 → 환경이 달라도 같은 해 (1단계 최적값은 불변).
solver   : PuLP/CBC. 상태가 Optimal이 아니면 최적해라고 주장하지 않는다.

출력: outputs/json/mclp_results.json, outputs/tables/mclp_selected_k{k}.csv,
      outputs/tables/mclp_coverage_curve.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import geopandas as gpd
import pulp

from src import config as C
from src.routing import dong_gu_map

SOLVER = pulp.PULP_CBC_CMD(msg=0)


def load_cover_data(radius_m: float = C.COVERAGE_RADIUS_M, mvi_scenario: str = C.MVI_PRIMARY) -> dict:
    dong = gpd.read_file(C.DONG_MASTER, encoding="utf-8")
    dong["dong_code"] = dong["dong_code"].astype(str)
    feat = pd.read_csv(C.DONG_FEATURES, encoding="utf-8-sig", dtype={"dong_code": str}).set_index("dong_code")
    mvi = pd.read_csv(C.PROCESSED_DIR / "mvi_results.csv", encoding="utf-8-sig", dtype={"dong_code": str})
    mvi = mvi.set_index("dong_code")

    bus = pd.read_csv(C.BUS_CLEAN, encoding="utf-8-sig")
    cand = bus[bus["join_status"] == "matched"].dropna(subset=["위도", "경도"]).reset_index(drop=True)
    if not (cand["위도"].between(*C.KOREA_LAT).all() and cand["경도"].between(*C.KOREA_LON).all()):
        raise ValueError("후보 정류장 좌표가 국내 범위를 벗어남")

    from src.network import dong_centroids_lonlat  # 수요 대표점(동 내 정류장 medoid)
    dlon, dlat = dong_centroids_lonlat(dong)
    dpt = gpd.GeoSeries(gpd.points_from_xy(dlon, dlat), crs=C.CRS_WGS84).to_crs(C.CRS_METRIC)
    cent = np.column_stack([dpt.x, dpt.y])
    cand_m = gpd.GeoSeries(gpd.points_from_xy(cand["경도"], cand["위도"]), crs=C.CRS_WGS84).to_crs(C.CRS_METRIC)
    cloc = np.column_stack([cand_m.x, cand_m.y])
    D = np.sqrt(((cloc[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2))
    codes = dong["dong_code"].to_numpy()
    elderly = feat.loc[codes, "elderly_population"].to_numpy(float)
    mvi_v = mvi.loc[codes, f"mvi_{mvi_scenario}"].to_numpy(float)
    gmap = dong_gu_map()
    return {"dong_codes": codes, "dong_names": dong["dong_name"].to_numpy(),
            "gu": np.array([gmap[c] for c in codes]),
            "elderly": elderly, "mvi": mvi_v, "vuln_w": elderly * mvi_v,
            "cover": D <= radius_m, "D": D, "cand": cand, "radius_m": radius_m,
            "mvi_scenario": mvi_scenario}


def tiebreak_cost(data: dict) -> np.ndarray:
    """후보별 '커버하는 동까지의 취약가중 평균거리'(m). 아무 동도 커버하지 않으면 반경값."""
    cov, D, w = data["cover"], data["D"], data["vuln_w"]
    num = (cov * D * w[None, :]).sum(axis=1)
    den = (cov * w[None, :]).sum(axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1), data["radius_m"])


def coverage_of(selected: list[int], data: dict) -> np.ndarray:
    cov = np.zeros(len(data["dong_codes"]), bool)
    for j in selected:
        cov |= data["cover"][j]
    return cov


def _two_stage(prob: pulp.LpProblem, primary, x: list, data: dict, name: str) -> tuple[str, float | None, list[int]]:
    """1단계 최적화 후 최적값을 고정하고 동점 해소 비용을 최소화."""
    prob.solve(SOLVER)
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return status, None, [j for j, v in enumerate(x) if v.value() and v.value() > 0.5]
    opt = float(pulp.value(primary))
    stage2 = pulp.LpProblem(f"{name}_tiebreak", pulp.LpMinimize)
    tc = tiebreak_cost(data)
    stage2 += pulp.lpSum(float(tc[j]) * x[j] for j in range(len(x)))
    for c in prob.constraints.values():
        stage2 += c
    stage2 += primary >= opt - 1e-6 * max(1.0, abs(opt))
    stage2.solve(SOLVER)
    status2 = pulp.LpStatus[stage2.status]
    if status2 != "Optimal":
        raise RuntimeError(f"{name}: 동점 해소 단계 비최적({status2}) — 1단계 최적값 {opt}")
    sel = [j for j, v in enumerate(x) if v.value() and v.value() > 0.5]
    if pulp.value(primary) < opt - 1e-5 * max(1.0, abs(opt)):
        raise RuntimeError(f"{name}: 동점 해소 후 1단계 목적값 하락")
    return "Optimal", opt, sel


def solve_mclp(data: dict, k: int) -> dict:
    Ncand, Ndong = data["cover"].shape
    prob = pulp.LpProblem("MCLP", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(Ncand)]
    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(Ndong)]
    obj = pulp.lpSum(float(data["vuln_w"][i]) * y[i] for i in range(Ndong))
    prob += obj
    prob += pulp.lpSum(x) == k
    for i in range(Ndong):
        cov_i = np.flatnonzero(data["cover"][:, i])
        prob += (y[i] <= pulp.lpSum(x[j] for j in cov_i)) if len(cov_i) else (y[i] == 0)
    status, opt, sel = _two_stage(prob, obj, x, data, f"MCLP_k{k}")
    return {"k": k, "status": status, "optimal": status == "Optimal", "selected_idx": sel,
            "covered": coverage_of(sel, data), "objective": opt}


def greedy_select(data: dict, k: int) -> list[int]:
    """고전적 탐욕 휴리스틱: 매 단계 추가 취약 커버가 최대인 정류장 (동점은 동점해소 비용 작은 순)."""
    tc = tiebreak_cost(data)
    sel: list[int] = []
    cov = np.zeros(len(data["dong_codes"]), bool)
    for _ in range(k):
        gain = (data["cover"] & ~cov[None, :]).astype(float) @ data["vuln_w"]
        gain[sel] = -1
        best = np.flatnonzero(gain == gain.max())
        j = int(best[np.argmin(tc[best])])
        sel.append(j)
        cov |= data["cover"][j]
    return sel


def top_demand_select(data: dict, k: int) -> list[int]:
    """단순 우선순위: 취약수요(고령×MVI) 상위 동부터 수요 대표점 최근접 정류장 1개씩."""
    order = np.argsort(-data["vuln_w"], kind="stable")[:k]
    sel: list[int] = []
    for i in order:
        d = data["D"][:, i].copy()
        d[sel] = np.inf
        sel.append(int(np.argmin(d)))
    return sel


def run_optimization(k_values: list[int] | None = None, radius_m: float = C.COVERAGE_RADIUS_M) -> dict:
    k_values = k_values or C.MCLP_K_VALUES
    data = load_cover_data(radius_m)
    tv, te = float(data["vuln_w"].sum()), float(data["elderly"].sum())
    summary = {"radius_m": radius_m, "n_candidates": int(data["cover"].shape[0]),
               "n_demand_points": int(data["cover"].shape[1]),
               "demand": f"65세 이상 인구 × MVI({C.MVI_PRIMARY})",
               "candidates": "행정동 매칭 검증된 실존 버스정류장",
               "tie_break": "1단계 최적값 고정 후 취약가중 평균 커버거리 최소화", "results": {}}
    curve = []
    for k in sorted(set(k_values) | set(C.COVERAGE_CURVE_K)):
        r = solve_mclp(data, k)
        if not r["optimal"]:
            raise RuntimeError(f"MCLP k={k} 비최적({r['status']})")
        cov = r["covered"]
        vc, ec = float(data["vuln_w"][cov].sum() / tv), float(data["elderly"][cov].sum() / te)
        curve.append({"k": k, "vuln_coverage": round(vc, 4), "elderly_coverage": round(ec, 4),
                      "n_covered_dongs": int(cov.sum())})
        if k in k_values:
            data["cand"].iloc[r["selected_idx"]].to_csv(C.TABLES / f"mclp_selected_k{k}.csv", index=False,
                                                        encoding="utf-8-sig")
            summary["results"][str(k)] = {
                "status": r["status"], "optimal": r["optimal"], "objective": round(r["objective"], 3),
                "vuln_coverage": round(vc, 4), "elderly_coverage": round(ec, 4),
                "n_covered_dongs": int(cov.sum()), "covered_dongs": sorted(data["dong_codes"][cov].tolist()),
                "selected_stop_ids": [str(v) for v in data["cand"].iloc[r["selected_idx"]]["정류장번호(ID)"]]}
            print(f"MCLP k={k}: status={r['status']} vuln_cov={vc:.3f} elder_cov={ec:.3f} dongs={int(cov.sum())}")
    cdf = pd.DataFrame(curve)
    cdf["marginal_gain"] = cdf["vuln_coverage"].diff().fillna(cdf["vuln_coverage"]).round(4)
    cdf.to_csv(C.TABLES / "mclp_coverage_curve.csv", index=False, encoding="utf-8-sig")
    summary["coverage_curve"] = cdf.to_dict("records")
    (C.JSON_DIR / "mclp_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    return summary


if __name__ == "__main__":
    run_optimization()
