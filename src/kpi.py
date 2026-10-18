"""공통 KPI — baseline·정책 A/B/C·What-if가 모두 이 함수로 평가된다 (동일 KPI 원칙).

커버리지류는 0~1 비율, 인구는 명, 거리는 km/m, 시간은 분.
"""
from __future__ import annotations

import json

import numpy as np

from src import config as C
from src.optimization import coverage_of


def reference_sets(data: dict) -> dict:
    """Robust Hotspot 집합과 MVI 상위 k 동 집합. 파일이 없으면 예외 (0으로 대체하지 않음)."""
    meta_p = C.TABLES / "robust_hotspot_meta.json"
    if not meta_p.exists():
        raise FileNotFoundError(f"{meta_p} 없음 — 공간분석 단계를 먼저 실행해야 KPI(hotspot 커버리지) 계산 가능")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    order = np.argsort(-data["mvi"], kind="stable")[:C.TOP_K_VULNERABLE]
    return {"hotspot": set(meta["dong_codes"]), "top": set(data["dong_codes"][order])}


def coverage_kpis(selected: list[int], data: dict, refs: dict | None = None) -> dict:
    refs = refs or reference_sets(data)
    cov = coverage_of(selected, data)
    w, e = data["vuln_w"], data["elderly"]
    is_hot = np.isin(data["dong_codes"], list(refs["hotspot"]))
    is_top = np.isin(data["dong_codes"], list(refs["top"]))
    gu_cov = {}
    for g in sorted(set(data["gu"])):
        m = data["gu"] == g
        gu_cov[g] = round(float(w[m & cov].sum() / w[m].sum()), 4)
    wn = w / w.sum()
    mean_c = float((cov * wn).sum())
    return {
        "n_new_stops": len(selected),
        "vuln_coverage": round(float(w[cov].sum() / w.sum()), 4),
        "elderly_coverage": round(float(e[cov].sum() / e.sum()), 4),
        "covered_vulnerable_demand": round(float(w[cov].sum()), 1),
        "covered_elderly": int(e[cov].sum()),
        f"top{C.TOP_K_VULNERABLE}_coverage": round(float(is_top[cov].sum() / is_top.sum()), 4),
        "hotspot_coverage": round(float(w[cov & is_hot].sum() / w[is_hot].sum()), 4) if is_hot.any() else None,
        "n_covered_dongs": int(cov.sum()),
        "uncovered_vulnerable_demand": round(float(w[~cov].sum()), 1),
        "min_gu_coverage": min(gu_cov.values()),
        "max_gu_coverage": max(gu_cov.values()),
        "gu_coverage_gap": round(max(gu_cov.values()) - min(gu_cov.values()), 4),
        "gu_coverages": gu_cov,
        "coverage_std_weighted": round(float(np.sqrt(((cov - mean_c) ** 2 * wn).sum())), 4),
    }


def route_kpis(cov_k: dict, plan_dict: dict) -> dict:
    km = plan_dict["total_vehicle_km"]
    return {"total_vehicle_km": km, "max_cycle_min": plan_dict["max_cycle_min"],
            "vehicles_used": plan_dict["vehicles_used"],
            "covered_elderly_per_km": round(cov_k["covered_elderly"] / km, 1) if km > 0 else None,
            "vuln_coverage_per_10km": round(cov_k["vuln_coverage"] / km * 10, 4) if km > 0 else None}
