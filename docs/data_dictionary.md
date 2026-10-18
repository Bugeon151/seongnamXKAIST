# 데이터 사전 (data_dictionary) — 주요 산출물

## data/SHA256SUMS
- 원본(`data/raw/**`)과 지오코딩 입력(`data/processed/*_geocoded.csv`)의 기준 해시. 한 줄 `<SHA-256>  <상대경로>`,
  해시는 줄바꿈(CRLF→LF) 정규화 후 계산. 데이터 검증 1단계(`raw_sha256_manifest`)가 매 실행 대조한다.

## data/processed (전처리 산출, `src/preprocessing.py`가 raw에서 재생성)
- `population_clean.csv` 동 50행: dong_code, dong_name, region_full_name, total_population, elderly_population(기준월), 월별 원자료
- `bus_stops_clean.csv` 정류장 1,304행 + official_dong_code/name, join_status(matched/outside_polygon/no_coordinates)
- `medical_facilities_clean.csv` 1,904행 + unique_facility_id(MED_xxxxx), facility_types
- `care_facilities_clean.csv` 301행: care_id, svc_* 급여 플래그, facility_based, capacity(정원), reference_date
- `dong_features_base.csv`, `seongnam_dong_master.geojson` 동별 인구·면적·고령비율·밀도·정류장 수(+경계)
- `demand_points.csv` 동 수요 대표점(정류장 medoid), 폴리곤 중심점, 이동거리 shift_m
- `accessibility.csv` 아래 accessibility_by_dong과 동일, `mvi_results.csv` 시나리오별 MVI·순위
- `*_geocoded.csv` (외부 API 실측, 버전관리 입력): 좌표, geocode_status/attempt, in_seongnam, 반환주소

## outputs/tables
- `accessibility_by_dong.csv`: med_nearest_network_m, med_nearest_euclid_m, med_within_1km_network,
  med_e2sfca_network(시설/고령 1천명), med_e2sfca_euclid, med_e2sfca_S2_network, wel_e2sfca_network(정원석/고령 1천명),
  wel_nearest_network_m, wel_capacity_in_catchment, transit_stops_per_1000_elderly, bus_nearest_network_m
- `accessibility_method_comparison.csv`: rank_nearest, rank_e2sfca_network, rank_e2sfca_euclid, 순위 이동
- `mvi_components.csv`: need_*, need, access_medical/welfare/transit, access, access_deficit, mvi, mvi_rank, weakest_access
- `mvi_ablation.csv`: 변형별 spearman_vs_primary, top10_overlap, jaccard_top
- `local_moran_results.csv` (150행): moran_I, moran_p, moran_quad, moran_class, significant, scenario
- `robustness_grid.csv` (36행), `hotspot_frequency.csv` (hh_share, top10_share, rank 통계, hotspot_class)
- `mclp_coverage_curve.csv` (k=1..10), `mclp_selected_k{3,5,7}.csv`, `baseline_comparison.csv`
- `policy_comparison.csv` (A/B/C × 목표·목적함수·제약·결정변수·KPI), `policy_{A,B,C}_selected.csv`
- `policy_C_pareto.csv` (예산별 route_km, vuln_coverage, nondominated, same_solution_as_budget, is_knee)
- `before_after_by_dong.csv` / `_by_gu.csv`: policy × metric(served_elderly, uncovered_vulnerable_demand,
  walk_to_service_m) × baseline/after/difference/percent_change
- `data_validation.csv`: stage(raw/processed), check, level(PASS/WARN/FAIL), detail

## outputs/json
- `dong_results.json` 동별 evidence(JSON Schema: `evidence_schema.json`), `explanations.json` 설명·생성기·시도 이력
- `grounding_validation.json`, `grounding_failure_demo.json`, `policy_results.json`, `policy_C_pareto.json`
- `mclp_results.json`, `baseline_summary.json`, `accessibility_summary.json`, `mvi_summary.json`, `spatial_summary.json`
- `llm_grounding_check.json` (선택: 실제 로컬 LLM 측정 기록)

## KPI 정의 (`src/kpi.py`)
vuln_coverage = 커버 동 취약수요(고령×MVI) 합 / 전체, elderly_coverage, top10_coverage(MVI 상위 10개 동 중 커버 비율),
hotspot_coverage, min_gu_coverage·gu_coverage_gap(구별 취약커버율 최소·최대-최소), total_vehicle_km(1회 순환),
max_cycle_min, vehicles_used, vuln_coverage_per_10km, covered_elderly_per_km, mean_walk_to_service_m_covered.
