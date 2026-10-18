"""통합·파이프라인 테스트 — raw부터 새로 실행한 결과(outputs_validation/_pytest)의 불변식 검증.

'값이 존재한다'가 아니라 수학적으로 성립해야 하는 관계(최적성 부등식, 공급 보존, KPI 재계산 일치,
Before/After 산술, 발표 수치=원천 필드)를 확인한다.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config as C
from tests.conftest import load_json

pytestmark = pytest.mark.usefixtures("pipeline_run")


def _csv(root, rel):
    return pd.read_csv(root / "outputs" / rel, encoding="utf-8-sig", dtype={"dong_code": str})


def test_all_stages_pass(pipeline_run):
    st = _csv(pipeline_run, "logs/pipeline_status.csv")
    assert len(st) == 8 and (st["status"] == "PASS").all(), st.to_string()
    assert [s.split("] ")[1] for s in st["stage"]] == ["Data validation", "Preprocessing", "Accessibility", "MVI",
                                                        "Spatial analysis", "Optimization", "Explanation validation",
                                                        "Outputs"]
    meta = load_json(pipeline_run, "run_metadata.json")
    assert meta["seeds"] == {"MORAN_SEED": C.MORAN_SEED, "RANDOM_SEED": C.RANDOM_SEED}
    assert all(len(f["sha256"]) == 64 for f in meta["source_files"])


def test_data_validation_no_fail(pipeline_run):
    v = _csv(pipeline_run, "tables/data_validation.csv")
    assert set(v["stage"]) == {"raw", "processed"} and (v["level"] != "FAIL").all()


def test_accessibility_invariants(pipeline_run):
    from src.accessibility import load_inputs, band_weights
    acc = pd.read_csv(pipeline_run / "data/processed/accessibility.csv", encoding="utf-8-sig")
    assert len(acc) == C.N_DONG
    cols = ["med_e2sfca_network", "med_e2sfca_euclid", "wel_e2sfca_network", "transit_stops_per_1000_elderly"]
    assert np.isfinite(acc[cols].to_numpy()).all() and (acc[cols] >= 0).all().all()
    fin = np.isfinite(acc["med_nearest_network_m"])
    assert (acc.loc[fin, "med_nearest_network_m"] >= acc.loc[fin, "med_nearest_euclid_m"] - 1.0).all()
    # 공급 보존: Σ P_i·A_i = 캐치먼트 내 수요가 있는 시설 공급 합 (A는 1,000명당 단위로 저장)
    inp = load_inputs()
    mats = np.load(pipeline_run / "data" / "interim" / "distance_matrices.npz")
    reach = (band_weights(mats["med_network"]) > 0).any(axis=0)
    assert (acc["elderly_population"] * acc["med_e2sfca_network"] / 1000).sum() == pytest.approx(reach.sum())
    reach_w = (band_weights(mats["wel_network"]) > 0).any(axis=0)
    assert (acc["elderly_population"] * acc["wel_e2sfca_network"] / 1000).sum() == pytest.approx(
        inp.welfare["capacity"].to_numpy()[reach_w].sum())


def test_mvi_invariants(pipeline_run):
    comp = _csv(pipeline_run, "tables/mvi_components.csv")
    assert comp["mvi"].between(0, 1).all() and comp["need"].between(0, 1).all()
    assert np.allclose(comp["mvi"], (comp["need"] * (1 - comp["access"])).round(4), atol=2e-4)
    acc_cols = ["access_medical", "access_welfare", "access_transit"]
    assert np.allclose(comp["access"], comp[acc_cols].mean(axis=1), atol=2e-4)
    assert sorted(comp["mvi_rank"]) == sorted(comp["mvi"].rank(ascending=False, method="min").astype(int))
    abl = _csv(pipeline_run, "tables/mvi_ablation.csv")
    assert {"without_medical", "without_welfare", "without_transit", "additive_form"} <= set(abl["variant"])


def test_spatial_invariants(pipeline_run):
    lm = _csv(pipeline_run, "tables/local_moran_results.csv")
    assert len(lm) == 3 * C.N_DONG
    assert ((lm["moran_class"] != "Not Significant") == (lm["moran_p"] < C.MORAN_ALPHA)).all()
    sig = lm[lm["moran_class"] != "Not Significant"]
    assert (sig["moran_class"] == sig["moran_quad"]).all()
    s = load_json(pipeline_run, "json/spatial_summary.json")
    assert s["weights"]["islands"] == [] and s["weights"]["components"] == 1 and s["weights"]["symmetric"]
    freq = _csv(pipeline_run, "tables/hotspot_frequency.csv")
    meta = load_json(pipeline_run, "tables/robust_hotspot_meta.json")
    assert set(meta["dong_codes"]) == set(freq.loc[freq["hh_share"] >= C.ROBUST_STABLE_SHARE, "dong_code"])
    grid = _csv(pipeline_run, "tables/robustness_grid.csv")
    assert len(grid) == len(C.MVI_SCENARIOS) * len(C.DECAY_SCALE_VARIANTS) * len(C.ROBUST_FORMS) * len(C.ROBUST_NORMALIZATIONS)


def test_mclp_and_baselines(pipeline_run):
    m = load_json(pipeline_run, "json/mclp_results.json")
    assert all(v["optimal"] for v in m["results"].values())
    curve = _csv(pipeline_run, "tables/mclp_coverage_curve.csv").sort_values("k")
    assert (curve["vuln_coverage"].diff().dropna() >= -1e-9).all()
    b = _csv(pipeline_run, "tables/baseline_comparison.csv")
    for k in C.MCLP_K_VALUES:
        s = b[b["k"] == str(k)].set_index("method")["vuln_coverage"].astype(float)
        for other in ["Greedy", "Top_demand", "Random_mean"]:
            assert s["MCLP"] >= s[other] - 1e-9, (k, other)


def test_policy_objectives_really_differ(pipeline_run):
    pol = load_json(pipeline_run, "json/policy_results.json")["policies"]
    A, B, Cc = (pol[k]["kpis"] for k in "ABC")
    assert A["vuln_coverage"] >= B["vuln_coverage"] - 1e-9       # A가 최적화하는 지표
    assert B["min_gu_coverage"] >= A["min_gu_coverage"] - 1e-9   # B가 최적화하는 지표
    assert A["n_new_stops"] == B["n_new_stops"] == C.K_MAIN
    assert pol["A"]["objective"] != pol["B"]["objective"] != pol["C"]["objective"]
    assert "순환노선" in pol["C"]["decision_variables"]          # C는 노선을 함께 결정
    assert Cc["n_new_stops"] <= C.POLICY_C_MAX_STOPS
    assert Cc["vuln_coverage"] >= A["vuln_coverage"] - 1e-9       # C: A와 같은 효과 이상
    assert Cc["total_vehicle_km"] <= A["total_vehicle_km"] + 1e-6  # C: 그 중 최소 운행거리(A의 해도 후보에 포함)


def test_policy_c_pareto_nondominated(pipeline_run):
    p = load_json(pipeline_run, "json/policy_C_pareto.json")
    rows = [r for r in p["rows"] if r["route_km"] is not None]
    assert all(r["optimal"] for r in rows), [r["status"] for r in rows]
    assert p["all_nondominated"]
    cov = [r["vuln_coverage"] for r in sorted(rows, key=lambda r: r["budget_km"])]
    assert all(b >= a - 1e-9 for a, b in zip(cov, cov[1:]))
    assert all(r["route_km"] <= r["budget_km"] + 1e-6 for r in rows)


def test_kpi_consistency_recomputed(pipeline_run):
    pol = load_json(pipeline_run, "json/policy_results.json")["policies"]
    feat = pd.read_csv(pipeline_run / "data/processed/dong_features_base.csv", encoding="utf-8-sig",
                       dtype={"dong_code": str}).set_index("dong_code")
    mvi = pd.read_csv(pipeline_run / "data/processed/mvi_results.csv", encoding="utf-8-sig",
                      dtype={"dong_code": str}).set_index("dong_code")
    w = feat["elderly_population"] * mvi.loc[feat.index, f"mvi_{C.MVI_PRIMARY}"]
    for k, p in pol.items():
        cov = p["covered_dongs"]
        assert p["kpis"]["covered_elderly"] == int(feat.loc[cov, "elderly_population"].sum())
        assert p["kpis"]["vuln_coverage"] == pytest.approx(w.loc[cov].sum() / w.sum(), abs=1e-4)
        assert p["kpis"]["total_vehicle_km"] == pytest.approx(sum(r["km"] for r in p["routes"]), abs=0.01)
        assert all(r["cycle_min"] <= C.MAX_CYCLE_MIN + 1e-6 for r in p["routes"])
        assert {s for r in p["routes"] for s in r["stop_ids"]} == {s["stop_id"] for s in p["selected_stops"]}


def test_before_after_arithmetic(pipeline_run):
    ba = _csv(pipeline_run, "tables/before_after_by_dong.csv")
    assert np.allclose(ba["difference"], ba["after"] - ba["baseline"], equal_nan=True)
    pc = ba["baseline"] > 0
    assert np.allclose(ba.loc[pc, "percent_change"], (ba.loc[pc, "difference"] / ba.loc[pc, "baseline"] * 100).round(2))
    pol = load_json(pipeline_run, "json/policy_results.json")["policies"]
    served = ba[ba["metric"] == "served_elderly"].groupby("policy")["after"].sum()
    for k in "ABC":
        assert served[k] == pol[k]["kpis"]["covered_elderly"]


def test_grounding_outputs(pipeline_run):
    from jsonschema import validate
    from src.explanation import DONG_SCHEMA
    g = load_json(pipeline_run, "json/grounding_validation.json")
    assert g["summary"]["fail"] == 0 and g["summary"]["n"] == C.N_DONG + 3
    assert load_json(pipeline_run, "json/grounding_failure_demo.json")["demo_pass"]
    for rec in load_json(pipeline_run, "json/dong_results.json"):
        validate(rec, DONG_SCHEMA)


def test_presentation_numbers_trace_to_source(pipeline_run):
    key = load_json(pipeline_run, "presentation/12_key_numbers.json")
    assert len(list((pipeline_run / "outputs" / "presentation").iterdir())) == 12
    for name, e in key.items():
        src = pipeline_run / e["source"]
        if src.suffix == ".json" and not any(ch in e["field"] for ch in "()=") and e["field"] != "summary":
            v = json.loads(src.read_text(encoding="utf-8"))
            for part in e["field"].split("."):
                v = v[part]
            assert v == e["value"], name


def test_what_if_recomputes_and_reproduces(pipeline_run):
    from src.what_if import WhatIfParams, run_what_if
    pol = load_json(pipeline_run, "json/policy_results.json")["policies"]
    same = run_what_if(WhatIfParams(policy="A", k=C.K_MAIN))
    assert {s["stop_id"] for s in same["result"]["selected_stops"]} == {s["stop_id"] for s in pol["A"]["selected_stops"]}
    more = run_what_if(WhatIfParams(policy="A", k=C.K_MAIN + 2))
    assert more["result"]["kpis"]["n_new_stops"] == C.K_MAIN + 2
    assert more["result"]["kpis"]["vuln_coverage"] >= same["result"]["kpis"]["vuln_coverage"]
    with pytest.raises(ValueError):
        run_what_if(WhatIfParams(policy="Z"))


def test_dashboard_what_if_widget_changes_computation(pipeline_run):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(C.BASE_DIR / "dashboard" / "app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    at.slider(key="wi_k").set_value(3)
    at.button(key="wi_run").click().run()
    assert not at.exception, at.exception
    assert any(m.label == "거점 수" and str(m.value) == "3" for m in at.metric)


def test_policy_c_never_worse_than_a_and_reports_reduction(pipeline_run):
    pol = load_json(pipeline_run, "json/policy_results.json")["policies"]
    s = pol["C"]["solver"]
    # 축약 후보가 A의 해를 포함하지 않을 수 있음을 숨기지 않고 기록하고, A의 해와 비교해 C ≤ A를 보장
    assert isinstance(s["A_solution_in_reduced_candidates"], bool)
    assert s["enumerated_route_km"] <= s["A_solution_route_km_exact"] + 1e-6
    assert "전역최적 아님" in s["note"]


def test_routes_reconcile_with_time_model_and_exact_tsp(pipeline_run):
    """노선 KPI 재계산: 순환시간 = 거리/속도 + 정차×정류장 수, 단일 차량 노선은 순열 완전탐색 최단거리와 일치."""
    from itertools import permutations
    from src import network as N
    from src.routing import load_depots
    pol = load_json(pipeline_run, "json/policy_results.json")["policies"]
    bus = pd.read_csv(pipeline_run / "data/processed/bus_stops_clean.csv", encoding="utf-8-sig")
    bus["sid"] = bus["정류장번호(ID)"].astype(str)
    dep = load_depots().set_index("gu")
    drive = N.load("drive")
    for p in pol.values():
        for r in p["routes"]:
            n = len(r["stop_ids"])
            assert r["cycle_min"] == pytest.approx(r["km"] / C.VEHICLE_SPEED_KMH * 60 + C.DWELL_MIN_PER_STOP * n, abs=0.06)
            if p["vehicles_required_by_gu"][r["gu"]] != 1:
                continue
            st = bus.set_index("sid").loc[r["stop_ids"]]
            lon = np.r_[dep.loc[r["gu"], "lon"], st["경도"].to_numpy(float)]
            lat = np.r_[dep.loc[r["gu"], "lat"], st["위도"].to_numpy(float)]
            D = drive.distance_matrix(lon, lat, lon, lat)
            best = min(D[0, q[0] + 1] + sum(D[a + 1, b + 1] for a, b in zip(q[:-1], q[1:])) + D[q[-1] + 1, 0]
                       for q in permutations(range(n)))
            assert r["km"] == pytest.approx(best / 1000, abs=1e-3)


@pytest.mark.parametrize("k", [1, 2])
def test_mclp_real_data_matches_brute_force(pipeline_run, k):
    """실제 데이터(후보 정류장 전체)에서 k=1,2 MCLP 최적값 = 모든 조합 완전탐색 최댓값."""
    from src.optimization import load_cover_data, solve_mclp
    data = load_cover_data()
    w = data["vuln_w"]
    cov = data["cover"]
    single = cov.astype(float) @ w
    if k == 1:
        best = single.max()
    else:
        best = 0.0
        for j in range(cov.shape[0]):
            best = max(best, float(((cov[j][None, :] | cov[j + 1:]).astype(float) @ w).max(initial=0.0)))
    r = solve_mclp(data, k)
    assert r["optimal"] and float(w[r["covered"]].sum()) == pytest.approx(best)


def test_stage_failure_stops_pipeline_and_skips_rest(monkeypatch):
    """실패 주입: 1단계가 실패하면 원인을 기록하고 이후 7단계는 SKIPPED, 종료코드 1, 메타데이터 미기록."""
    import shutil
    import run_pipeline
    import src.data_validation as dv
    tmp_path = C.BASE_DIR / "outputs_validation" / "_pytest_failure"
    shutil.rmtree(tmp_path, ignore_errors=True)
    for name in ["MAPS", "TABLES", "JSON_DIR", "FIGURES", "LOGS", "PROCESSED_DIR", "INTERIM_DIR", "OUTPUTS"]:
        monkeypatch.setattr(C, name, tmp_path / name.lower())

    def boom():
        raise dv.DataValidationError("주입된 오류")
    monkeypatch.setattr(dv, "validate_raw", boom)
    assert run_pipeline.run_stages() == 1
    st = pd.read_csv(tmp_path / "logs" / "pipeline_status.csv", encoding="utf-8-sig")
    assert st["status"].tolist() == ["FAIL"] + ["SKIPPED"] * 7
    assert "주입된 오류" in st.loc[0, "detail"]
    assert not (tmp_path / "outputs" / "run_metadata.json").exists()
