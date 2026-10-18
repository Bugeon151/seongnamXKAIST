"""입지 baseline 비교 — 동일 k·동일 반경·동일 수요정의·동일 KPI 함수(src/kpi.py).

A. Existing: 기존 정류장 전체 (신규 0개·상한 맥락 — 동일 k 비교 대상 아님)
B. Random  : 무작위 k개, RANDOM_BASELINE_SEEDS개 seed (평균·표준편차·최댓값)
C. Top-demand: 취약수요(고령×MVI) 상위 동부터 최근접 정류장 1개씩 (직관적 우선순위 배치)
D. Greedy  : 추가 커버 최대 정류장을 순차 선택 (고전 휴리스틱)
E. MCLP    : 정수계획 최적해 (CBC Optimal 확인분)
MCLP ≥ 모든 동일 k 대안 은 최적성에서 따라오는 성질이며 tests에서 검증한다.
출력: outputs/tables/baseline_comparison.csv, outputs/json/baseline_summary.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config as C
from src.optimization import load_cover_data, solve_mclp, greedy_select, top_demand_select
from src.kpi import coverage_kpis, reference_sets

KPI_COLS = ["vuln_coverage", "elderly_coverage", f"top{C.TOP_K_VULNERABLE}_coverage", "hotspot_coverage",
            "n_covered_dongs", "min_gu_coverage"]


def run_baseline(k_values: list[int] | None = None, radius_m: float = C.COVERAGE_RADIUS_M) -> pd.DataFrame:
    k_values = k_values or C.MCLP_K_VALUES
    data = load_cover_data(radius_m)
    refs = reference_sets(data)
    rows = [{"method": "Existing_all_stops", "k": "all",
             **{c: coverage_kpis(list(range(len(data["cand"]))), data, refs)[c] for c in KPI_COLS},
             "note": "기존 정류장 전체(신규 0개) — 상한 맥락, 동일 k 비교 대상 아님"}]
    rng = np.random.default_rng(C.RANDOM_SEED)
    for k in k_values:
        m = solve_mclp(data, k)
        if not m["optimal"]:
            raise RuntimeError(f"MCLP k={k} 비최적")
        res = {"MCLP": coverage_kpis(m["selected_idx"], data, refs),
               "Greedy": coverage_kpis(greedy_select(data, k), data, refs),
               "Top_demand": coverage_kpis(top_demand_select(data, k), data, refs)}
        rand = [coverage_kpis(rng.choice(len(data["cand"]), size=k, replace=False).tolist(), data, refs)
                for _ in range(C.RANDOM_BASELINE_SEEDS)]
        for name, r in res.items():
            rows.append({"method": name, "k": k, **{c: r[c] for c in KPI_COLS},
                         "note": "CBC Optimal" if name == "MCLP" else ""})
        rows.append({"method": "Random_mean", "k": k,
                     **{c: round(float(np.mean([r[c] for r in rand])), 4) for c in KPI_COLS},
                     "random_sd_vuln": round(float(np.std([r["vuln_coverage"] for r in rand])), 4),
                     "random_max_vuln": round(float(max(r["vuln_coverage"] for r in rand)), 4),
                     "note": f"{C.RANDOM_BASELINE_SEEDS} seeds (seed={C.RANDOM_SEED})"})
        for name in ["Top_demand", "Greedy"]:
            rows.append({"method": f"MCLP_minus_{name}_pp", "k": k,
                         "vuln_coverage": round((res["MCLP"]["vuln_coverage"] - res[name]["vuln_coverage"]) * 100, 2),
                         "note": "%p (동일 k)"})
        print(f"k={k}: MCLP={res['MCLP']['vuln_coverage']:.3f} greedy={res['Greedy']['vuln_coverage']:.3f} "
              f"top={res['Top_demand']['vuln_coverage']:.3f} rand={np.mean([r['vuln_coverage'] for r in rand]):.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(C.TABLES / "baseline_comparison.csv", index=False, encoding="utf-8-sig")
    piv = df[df["method"].isin(["MCLP", "Greedy", "Top_demand", "Random_mean"])].pivot(
        index="k", columns="method", values="vuln_coverage")
    summary = {"metric": "vuln_coverage (취약가중 고령수요 커버 비율)", "radius_m": radius_m,
               "by_k": {str(k): {m: float(piv.loc[k, m]) for m in piv.columns} for k in piv.index},
               "mclp_minus_top_demand_pp": {str(k): round((piv.loc[k, "MCLP"] - piv.loc[k, "Top_demand"]) * 100, 2)
                                            for k in piv.index},
               "mclp_minus_greedy_pp": {str(k): round((piv.loc[k, "MCLP"] - piv.loc[k, "Greedy"]) * 100, 2)
                                        for k in piv.index}}
    (C.JSON_DIR / "baseline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    return df


if __name__ == "__main__":
    run_baseline()
