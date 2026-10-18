"""MobiLens Seongnam 대시보드 (Streamlit).

실행: streamlit run dashboard/app.py   (먼저 python run_pipeline.py 로 결과 생성)
모든 화면 값은 파이프라인 산출물(outputs/)에서 읽고, What-if 탭만 src.what_if로 실제 재계산한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C

st.set_page_config(page_title="MobiLens Seongnam", layout="wide")
POLICY_COLOR = {"A": "#d95f02", "B": "#1b9e77", "C": "#7570b3"}
METRIC_LABEL = {"mvi": "MVI (Need × (1−Access))", "need": "Need (수요)", "access": "Access (종합 접근성)",
                "access_medical": "의료 접근성(정규화)", "access_welfare": "복지 접근성(정규화)",
                "access_transit": "대중교통 접근성(정규화)"}


@st.cache_data(show_spinner=False)
def load_all(out_root: str) -> dict:
    o = Path(out_root) / "outputs"
    req = ["run_metadata.json", "json/dong_results.json", "json/policy_results.json", "tables/mvi_components.csv"]
    missing = [r for r in req if not (o / r).exists()]
    if missing:
        raise FileNotFoundError(f"파이프라인 결과 없음: {missing} — 먼저 `python run_pipeline.py` 실행")
    rj = lambda r: json.loads((o / r).read_text(encoding="utf-8"))  # noqa: E731
    rc = lambda r: pd.read_csv(o / r, encoding="utf-8-sig", dtype={"dong_code": str})  # noqa: E731
    import geopandas as gpd
    dong = gpd.read_file(Path(out_root) / "data" / "processed" / "seongnam_dong_master.geojson")
    dong["dong_code"] = dong["dong_code"].astype(str)
    return {"meta": rj("run_metadata.json"), "dong_geo": json.loads(dong[["dong_code", "dong_name", "geometry"]].to_json()),
            "bounds": dong.total_bounds.tolist(), "recs": rj("json/dong_results.json"),
            "pol": rj("json/policy_results.json"), "comp": rc("tables/mvi_components.csv"),
            "freq": rc("tables/hotspot_frequency.csv"), "moran": rc("tables/local_moran_results.csv"),
            "rob": rj("tables/robustness_report.json"), "hot": rj("tables/robust_hotspot_meta.json"),
            "curve": rc("tables/mclp_coverage_curve.csv"), "base": rc("tables/baseline_comparison.csv"),
            "pareto": rc("tables/policy_C_pareto.csv"), "ba_gu": rc("tables/before_after_by_gu.csv"),
            "exps": rj("json/explanations.json"), "grd": rj("json/grounding_validation.json"),
            "demo": rj("json/grounding_failure_demo.json"), "val": rc("tables/data_validation.csv"),
            "acc_sum": rj("json/accessibility_summary.json"), "acc_cmp": rc("tables/accessibility_method_comparison.csv"),
            "key": rj("presentation/12_key_numbers.json"), "prob": rj("presentation/01_problem_summary.json"),
            "abl": rc("tables/mvi_ablation.csv"), "base_sum": rj("json/baseline_summary.json")}


def seongnam_map(d: dict, df: pd.DataFrame, col: str, title: str, scale="YlOrRd", hover=None, discrete=None):
    x0, y0, x1, y1 = d["bounds"]
    kw = dict(geojson=d["dong_geo"], locations="dong_code", featureidkey="properties.dong_code",
              center={"lat": (y0 + y1) / 2, "lon": (x0 + x1) / 2}, zoom=10.6, map_style="open-street-map",
              opacity=0.75, hover_name="dong_name", hover_data=hover or {})
    if discrete:
        fig = px.choropleth_map(df, color=col, color_discrete_map=discrete, **kw)
    else:
        fig = px.choropleth_map(df, color=col, color_continuous_scale=scale, **kw)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=520, title=title)
    return fig


def add_policy_layers(fig, pol: dict, keys) -> None:
    for k in keys:
        p = pol["policies"][k]
        for r in p["routes"]:
            if r.get("path_lonlat"):
                fig.add_trace(go.Scattermap(lon=[q[0] for q in r["path_lonlat"]], lat=[q[1] for q in r["path_lonlat"]],
                                            mode="lines", line=dict(width=3, color=POLICY_COLOR[k]),
                                            name=f"정책 {k} 노선({r['gu']}, {r['km']}km)", hoverinfo="name"))
        s = pd.DataFrame(p["selected_stops"])
        fig.add_trace(go.Scattermap(lon=s["lon"], lat=s["lat"], mode="markers",
                                    marker=dict(size=13, color=POLICY_COLOR[k]), name=f"정책 {k} 거점",
                                    text=s["name"] + " (" + s["dong"] + ")", hoverinfo="text"))


def main() -> None:
    st.title("MobiLens Seongnam — 교통약자 의료·복지 접근성 의사결정 지원")
    st.caption("AI는 정책을 결정하지 않습니다. 공개 집계자료로 계산한 근거와 정책 trade-off를 제시합니다.")
    try:
        d = load_all(str(C.OUT_ROOT))
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    m = d["meta"]
    st.caption(f"결과 생성 {m['finished_at']} · 코드 {C.METHOD_VERSION} · git {str(m.get('git_sha'))[:8]} · 인구 기준 {C.BASE_MONTH}")
    tabs = st.tabs(["개요", "이동취약성(MVI)", "Hotspot", "최적화", "정책 비교", "What-if", "설명·검증", "데이터 품질"])

    with tabs[0]:
        prob, key = d["prob"], d["key"]
        c = st.columns(4)
        c[0].metric("65세 이상 인구", f"{prob['elderly_population']:,}명", f"{prob['elderly_share_pct']}%")
        c[1].metric("MVI 1위", key["top_vulnerable_dong"]["value"], f"MVI {key['top_vulnerable_mvi']['value']:.3f}")
        c[2].metric("안정 Robust Hotspot", f"{prob['stable_robust_hotspots']}개 동")
        k5 = str(C.K_MAIN)
        c[3].metric(f"MCLP k={k5} 취약수요 커버", f"{d['base_sum']['by_k'][k5]['MCLP']:.3f}",
                    f"+{d['base_sum']['mclp_minus_top_demand_pp'][k5]}%p vs 상위동 우선")
        st.subheader(f"취약 상위 {C.TOP_K_VULNERABLE}개 동")
        top = d["comp"].nsmallest(C.TOP_K_VULNERABLE, "mvi_rank")[
            ["mvi_rank", "dong_name", "mvi", "need", "access", "weakest_access_label", "elderly_population", "elderly_ratio"]]
        st.dataframe(top.rename(columns={"mvi_rank": "순위", "dong_name": "행정동", "weakest_access_label": "가장 낮은 접근성",
                                         "elderly_population": "65세 이상(명)", "elderly_ratio": "고령비율(%)"}),
                     hide_index=True, use_container_width=True)
        st.plotly_chart(seongnam_map(d, d["comp"], "mvi", "성남시 MVI"), use_container_width=True)

    with tabs[1]:
        col = st.selectbox("지도 지표", list(METRIC_LABEL), format_func=METRIC_LABEL.get)
        l, r = st.columns([3, 2])
        fig = seongnam_map(d, d["comp"], col, METRIC_LABEL[col],
                           hover={"mvi": ":.3f", "mvi_rank": True, "need": ":.3f", "access": ":.3f"})
        ev = l.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="mvi_map")
        names = d["comp"].sort_values("mvi_rank")["dong_name"].tolist()
        picked = None
        pts = (ev or {}).get("selection", {}).get("points", []) if isinstance(ev, dict) or hasattr(ev, "get") else []
        if pts:
            picked = d["comp"].loc[d["comp"]["dong_code"] == pts[0].get("location"), "dong_name"].tolist()
        name = r.selectbox("행정동 (지도 클릭 또는 선택)", names, index=names.index(picked[0]) if picked else 0)
        row = d["comp"][d["comp"]["dong_name"] == name].iloc[0]
        r.metric("MVI", f"{row['mvi']:.3f}", f"{int(row['mvi_rank'])}위 / {C.N_DONG}")
        r.bar_chart(pd.Series({"Need": row["need"], "의료": row["access_medical"], "복지": row["access_welfare"],
                               "대중교통": row["access_transit"], "1−Access": row["access_deficit"]}), height=220)
        r.write(f"가장 낮은 접근성: **{row['weakest_access_label']}** · 65세 이상 {int(row['elderly_population']):,}명"
                f" ({row['elderly_ratio']}%)")
        r.caption("구성요소는 50개 동 기준 min-max 정규화(0=최저, 1=최고). MVI = Need × (1 − 세 접근성 평균).")
        st.subheader("구성요소 제거 실험 (ablation)")
        st.dataframe(d["abl"], hide_index=True, use_container_width=True)

    with tabs[2]:
        f = d["freq"].copy()
        f["분류"] = f["hotspot_class"].map({"stable": "안정 hotspot", "unstable": "불안정 hotspot", "none": "해당 없음"})
        st.plotly_chart(seongnam_map(d, f, "분류", "Robust Hotspot (강건성 격자 출현 빈도 기준)",
                                     discrete={"안정 hotspot": "#b2182b", "불안정 hotspot": "#ef8a62", "해당 없음": "#dddddd"},
                                     hover={"hh_share": ":.2f", "rank_mean": ":.1f"}), use_container_width=True)
        rob = d["rob"]
        c = st.columns(4)
        c[0].metric("격자 시나리오", rob["grid"]["n_scenarios"])
        c[1].metric("순위 상관(최소, vs primary)", rob["rank_correlation"]["vs_primary_min"])
        c[2].metric("HH 집합 Jaccard(평균)", rob["jaccard_HH"]["vs_primary_mean"])
        c[3].metric("안정 / 불안정", f"{len(rob['stable_hotspots'])} / {len(rob['unstable_hotspots'])}")
        st.write("primary 한 시나리오만 보면 hotspot이지만 안정적이지 않은 동:",
                 ", ".join(rob["no_robustness_vs_robust"]["primary_HH_not_stable"]) or "없음")
        st.dataframe(f[f["hh_share"] > 0][["dong_name", "hh_share", f"top{C.TOP_K_VULNERABLE}_share", "rank_mean",
                                           "rank_sd", "분류"]], hide_index=True, use_container_width=True)

    with tabs[3]:
        l, r = st.columns(2)
        cv = d["curve"]
        fig = go.Figure([go.Scatter(x=cv["k"], y=cv["vuln_coverage"], mode="lines+markers", name="취약수요 커버"),
                         go.Bar(x=cv["k"], y=cv["marginal_gain"], name="한계효과", opacity=0.4)])
        fig.update_layout(title="MCLP 커버리지 곡선 (각 k 최적해)", xaxis_title="거점 수 k", yaxis_title="비율", height=380)
        l.plotly_chart(fig, use_container_width=True)
        b = d["base"][d["base"]["method"].isin(["MCLP", "Greedy", "Top_demand", "Random_mean"])]
        fig = px.bar(b, x="k", y="vuln_coverage", color="method", barmode="group", title="입지 방법 비교 (동일 k·반경)")
        fig.update_layout(height=380)
        r.plotly_chart(fig, use_container_width=True)
        fig = seongnam_map(d, d["comp"], "mvi", f"정책 A (MCLP k={C.K_MAIN}) 선정 거점", scale="Greys")
        add_policy_layers(fig, d["pol"], ["A"])
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        pol = d["pol"]
        rows = [{"정책": p["name"], "목표": p["goal"], "목적함수": p["objective"], "제약": p["constraints"],
                 **{k: p["kpis"][k] for k in ["n_new_stops", "vuln_coverage", f"top{C.TOP_K_VULNERABLE}_coverage",
                                              "min_gu_coverage", "gu_coverage_gap", "total_vehicle_km", "max_cycle_min",
                                              "vehicles_used", "vuln_coverage_per_10km"]}} for p in pol["policies"].values()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.info(pol["tradeoff"])
        l, r = st.columns(2)
        pa = d["pareto"][d["pareto"]["route_km"].notna()].sort_values("route_km")
        fig = go.Figure([go.Scatter(x=pa["route_km"], y=pa["vuln_coverage"], mode="lines+markers",
                                    name="정책 C Pareto", line=dict(color=POLICY_COLOR["C"]))])
        for k, p in pol["policies"].items():
            fig.add_trace(go.Scatter(x=[p["kpis"]["total_vehicle_km"]], y=[p["kpis"]["vuln_coverage"]], mode="markers+text",
                                     text=[k], textposition="top center", marker=dict(size=16, color=POLICY_COLOR[k]),
                                     name=f"정책 {k}"))
        fig.update_layout(title="운행거리 대비 커버리지 trade-off", xaxis_title="vehicle-km", yaxis_title="취약수요 커버",
                          height=420)
        l.plotly_chart(fig, use_container_width=True)
        g = d["ba_gu"][d["ba_gu"]["metric"] == "served_elderly"]
        fig = px.bar(g, x="gu", y="after", color="policy", barmode="group", color_discrete_map=POLICY_COLOR,
                     title="Before(0) → After: 구별 서비스 반경 내 고령인구")
        fig.update_layout(height=420)
        r.plotly_chart(fig, use_container_width=True)
        show = st.multiselect("지도에 표시할 정책", ["A", "B", "C"], default=["A", "B", "C"])
        fig = seongnam_map(d, d["comp"], "mvi", "정책별 거점·노선", scale="Greys")
        add_policy_layers(fig, pol, show)
        st.plotly_chart(fig, use_container_width=True)
        st.json(pol["uniqueness_check"], expanded=False)

    with tabs[5]:
        st.write("파라미터를 바꾸고 **재계산**을 누르면 최적화·노선·KPI를 실제로 다시 계산합니다.")
        c = st.columns(4)
        policy = c[0].selectbox("정책", ["A", "B", "C"], key="wi_policy")
        k = c[1].slider("거점 수 k (C는 최대)", 1, 12, C.K_MAIN, key="wi_k")
        radius = c[2].slider("서비스 반경(m)", 500, 3000, int(C.COVERAGE_RADIUS_M), 100, key="wi_radius")
        budget = c[3].slider("C 운행거리 예산(km)", 5, 100, 30, 5, key="wi_budget")
        c = st.columns(4)
        scen = c[0].selectbox("MVI 가중치", list(C.MVI_SCENARIOS), key="wi_scen")
        veh = c[1].number_input("구별 최소 차량 수", 1, 5, C.VEHICLES_PER_DEPOT, key="wi_veh")
        cyc = c[2].slider("최대 순환시간(분)", 30, 180, int(C.MAX_CYCLE_MIN), 10, key="wi_cycle")
        spd = c[3].slider("평균 속도(km/h)", 10, 40, int(C.VEHICLE_SPEED_KMH), key="wi_speed")
        if st.button("재계산", type="primary", key="wi_run"):
            from src.what_if import WhatIfParams, run_what_if
            with st.spinner("최적화·노선 재계산 중..."):
                try:
                    st.session_state["wi"] = run_what_if(WhatIfParams(policy, int(k), float(budget), float(radius), scen,
                                                                      int(veh), float(cyc), float(spd)))
                except (ValueError, RuntimeError, FileNotFoundError) as e:
                    st.session_state["wi"] = {"error": str(e)}
        wi = st.session_state.get("wi")
        if wi and "error" in wi:
            st.error(wi["error"])
        elif wi and "result" not in wi:
            st.warning(f"{wi['message']} (status={wi['status']})")
        elif wi:
            kp, base = wi["result"]["kpis"], d["pol"]["policies"][wi["params"]["policy"]]["kpis"]
            c = st.columns(5)
            c[0].metric("거점 수", kp["n_new_stops"], kp["n_new_stops"] - base["n_new_stops"])
            c[1].metric("취약수요 커버", f"{kp['vuln_coverage']:.3f}", f"{kp['vuln_coverage'] - base['vuln_coverage']:+.3f}")
            c[2].metric("최소 구 커버", f"{kp['min_gu_coverage']:.3f}", f"{kp['min_gu_coverage'] - base['min_gu_coverage']:+.3f}")
            c[3].metric("운행거리(km)", f"{kp['total_vehicle_km']:.1f}", f"{kp['total_vehicle_km'] - base['total_vehicle_km']:+.1f}")
            c[4].metric("차량(대)", kp["vehicles_used"], kp["vehicles_used"] - base["vehicles_used"])
            st.caption(f"solver status: {wi['status']} · 기준값(delta)은 파이프라인 정책 {wi['params']['policy']} 결과")
            fig = seongnam_map(d, d["comp"], "mvi", "What-if 결과 거점·노선", scale="Greys")
            add_policy_layers(fig, {"policies": {wi["params"]["policy"]: wi["result"]}}, [wi["params"]["policy"]])
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pd.DataFrame(wi["before_after_by_gu"]), hide_index=True, use_container_width=True)

    with tabs[6]:
        exps = d["exps"]
        opts = [f"[{e['kind']}] {e['evidence'].get('region_name') or e['evidence'].get('name')}" for e in exps]
        i = st.selectbox("설명 대상", range(len(opts)), format_func=lambda j: opts[j])
        e = exps[i]
        from src.grounding_validator import validate_explanation
        v = validate_explanation(e["final_text"], e["evidence"])
        (st.success if v["FINAL"] == "PASS" else st.error)(f"grounding {v['FINAL']} · 생성기: {e['generator']}")
        st.write(e["final_text"])
        l, r = st.columns(2)
        l.caption("근거 JSON (LLM 입력의 전부)")
        l.json(e["evidence"], expanded=False)
        r.caption("검증 상세")
        r.json({"numeric": v["numeric"], "places": v["entity"], "policies": v["policy_mentions"],
                "violations": v["violations"], "attempts": e["attempts"]}, expanded=False)
        st.subheader("고의 오류 탐지 시연")
        st.dataframe(pd.DataFrame([{"case": k2, "FINAL": x["FINAL"], "사유": "; ".join(x["violations"])}
                                   for k2, x in d["demo"].items() if isinstance(x, dict) and "FINAL" in x]),
                     hide_index=True, use_container_width=True)
        st.caption(f"전체 최종 설명 grounding: {d['grd']['summary']['pass']}/{d['grd']['summary']['n']} PASS "
                   f"{d['grd']['summary']['by_generator']}")

    with tabs[7]:
        val = d["val"]
        c = st.columns(3)
        for j, lv in enumerate(["PASS", "WARN", "FAIL"]):
            c[j].metric(lv, int((val["level"] == lv).sum()))
        st.dataframe(val, hide_index=True, use_container_width=True)
        st.subheader("원천 데이터 (SHA-256·기준일)")
        st.dataframe(pd.DataFrame(m["source_files"]), hide_index=True, use_container_width=True)
        p = d["prob"]
        st.write(f"의료기관 지오코딩 성남시 내 {p['medical_facilities_geocoded_in_seongnam']}/{p['medical_facilities_total']} · "
                 f"재가요양기관 {p['welfare_geocoded_in_seongnam']}/{p['welfare_total']} (Nominatim 실측, 임의 좌표 없음)")
        st.write("공급량 정의:", d["acc_sum"]["supply"])
        st.warning("한계: 의료 공급량은 시설 수 proxy(병상·의사수 없음), 수요점은 행정동 대표점(동 내 정류장 medoid), "
                   "서비스 반경은 직선거리, "
                   "노선 속도·정차시간은 가정값, OSM 도로망 스냅샷 기준. 상세는 docs/limitations.md")


main()
