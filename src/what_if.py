"""What-if — 주요 파라미터를 바꿔 입지·노선·KPI·Before/After를 실제로 재계산한다.

대시보드와 CLI(run_pipeline.py --what-if)가 이 함수를 호출한다. 저장된 결과를 조회하지 않고
매번 최적화(MCLP/Maximin/Orienteering)와 노선 모형을 다시 푼다.

입력 파라미터(WhatIfParams):
  policy        'A' | 'B' | 'C'
  k             정류장 수 (A/B), C는 최대 정류장 수
  budget_km     C의 노선거리 예산
  radius_m      서비스 반경(커버 기준)
  mvi_scenario  수요 가중에 쓸 MVI 가중치 시나리오
  vehicles_per_depot, max_cycle_min, speed_kmh, dwell_min   노선 운영 가정
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from src import config as C
from src.optimization import load_cover_data, solve_mclp
from src.policy_scenarios import (solve_maximin, reduce_candidates, seed_candidates, solve_orienteering,
                                   evaluate_policy, before_after)
from src.routing import RouteParams
from src.kpi import reference_sets

MAX_K = 12


@dataclass
class WhatIfParams:
    policy: str = "A"
    k: int = C.K_MAIN
    budget_km: float = 30.0
    radius_m: float = C.COVERAGE_RADIUS_M
    mvi_scenario: str = C.MVI_PRIMARY
    vehicles_per_depot: int = C.VEHICLES_PER_DEPOT
    max_cycle_min: float = C.MAX_CYCLE_MIN
    speed_kmh: float = C.VEHICLE_SPEED_KMH
    dwell_min: float = C.DWELL_MIN_PER_STOP

    def validate(self) -> None:
        if self.policy not in ("A", "B", "C"):
            raise ValueError("policy는 A/B/C 중 하나")
        if not 1 <= self.k <= MAX_K:
            raise ValueError(f"k는 1~{MAX_K}")
        if not 300 <= self.radius_m <= 5000:
            raise ValueError("radius_m은 300~5000m")
        if self.mvi_scenario not in C.MVI_SCENARIOS:
            raise ValueError(f"mvi_scenario는 {list(C.MVI_SCENARIOS)} 중 하나")
        if self.vehicles_per_depot < 1 or self.max_cycle_min <= 0 or self.speed_kmh <= 0 or self.dwell_min < 0:
            raise ValueError("노선 운영 파라미터가 유효하지 않음")
        if self.policy == "C" and self.budget_km <= 0:
            raise ValueError("budget_km > 0 필요")


def run_what_if(p: WhatIfParams) -> dict:
    p.validate()
    data = load_cover_data(p.radius_m, p.mvi_scenario)
    refs = reference_sets(data)
    rp = RouteParams(p.speed_kmh, p.dwell_min, p.vehicles_per_depot, p.max_cycle_min)
    if p.policy == "A":
        r = solve_mclp(data, p.k)
    elif p.policy == "B":
        r = solve_maximin(data, p.k)
    else:
        red = reduce_candidates(data, seeds=seed_candidates(data))
        r = solve_orienteering(data, red, p.budget_km, max_stops=p.k)
    if not r["selected_idx"]:
        return {"params": asdict(p), "status": r["status"], "optimal": r.get("optimal", False),
                "message": "선택된 정류장 없음 (예산이 너무 작거나 해 없음)"}
    ev = evaluate_policy(p.policy, r["selected_idx"], data, refs, rp, with_paths=True)
    ba, ba_gu = before_after(data, {p.policy: r["selected_idx"]})
    return {"params": asdict(p), "status": r["status"], "optimal": bool(r.get("optimal", False)),
            "result": ev, "before_after_by_dong": ba.to_dict("records"),
            "before_after_by_gu": ba_gu.to_dict("records")}
