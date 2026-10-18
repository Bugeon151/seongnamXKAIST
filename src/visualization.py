"""시각화 — 정적 지도·그래프(PNG) + 인터랙티브 지도(HTML). 모든 값은 앞 단계 산출물에서 읽는다.

지도는 성남시 50개 행정동 범위로 자동 맞춤. 각 그림 하단에 출처·기준일·방법을 표기한다.
"""
from __future__ import annotations

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import geopandas as gpd

from src import config as C

KOREAN_FONTS = ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans KR", "Noto Sans CJK KR"]
_font = None
for _f in KOREAN_FONTS:
    try:
        fm.findfont(_f, fallback_to_default=False)
        _font = _f
        break
    except ValueError:
        continue
if _font:
    plt.rcParams["font.family"] = _font
else:
    warnings.warn("한글 글꼴을 찾지 못함 — 그림의 한글이 깨질 수 있음 (NanumGothic 등 설치 권장)")
plt.rcParams["axes.unicode_minus"] = False

SRC_NOTE = "출처: 성남시 공공데이터(인구·버스·의료·재가요양), SGIS 기반 행정동 경계, © OpenStreetMap contributors"
TIME_NOTE = f"인구 기준 {C.BASE_MONTH}"
POLICY_COLOR = {"A": "#d95f02", "B": "#1b9e77", "C": "#7570b3"}


def _dong() -> gpd.GeoDataFrame:
    d = gpd.read_file(C.DONG_MASTER, encoding="utf-8")
    d["dong_code"] = d["dong_code"].astype(str)
    return d


def _finish(fig, ax, title: str, note: str, path) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.text(0.01, 0.005, f"{SRC_NOTE} | {TIME_NOTE} | {note}", fontsize=6.5, color="#444")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _map_axes(dong: gpd.GeoDataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 8))
    x0, y0, x1, y1 = dong.total_bounds
    ax.set_xlim(x0 - 0.005, x1 + 0.005)
    ax.set_ylim(y0 - 0.005, y1 + 0.005)
    ax.set_axis_off()
    return fig, ax


def mvi_map(dong, comp) -> None:
    g = dong.merge(comp[["dong_code", "mvi", "mvi_rank"]], on="dong_code")
    fig, ax = _map_axes(g)
    g.plot(column="mvi", cmap="YlOrRd", linewidth=0.6, edgecolor="white", legend=True, ax=ax,
           legend_kwds={"label": "MVI = Need × (1 - Access)  (0~1, 높을수록 취약)", "shrink": 0.6})
    for _, r in g.nsmallest(5, "mvi_rank").iterrows():
        p = r.geometry.representative_point()
        ax.annotate(f"{int(r.mvi_rank)}. {r.dong_name}", (p.x, p.y), fontsize=8, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, lw=0))
    _finish(fig, ax, "이동취약지수(MVI) — 상위 5개 동 표시", "접근성: OSM 보행망 E2SFCA(의료·복지)+정류장 밀도",
            C.MAPS / "mvi_map.png")


def moran_map(dong, moran) -> None:
    g = dong.merge(moran[moran["scenario"] == C.MVI_PRIMARY][["dong_code", "moran_class"]], on="dong_code")
    colors = {"High-High": "#d7191c", "Low-Low": "#2c7bb6", "High-Low": "#fdae61", "Low-High": "#abd9e9",
              "Not Significant": "#eeeeee"}
    fig, ax = _map_axes(g)
    for cls, col in colors.items():
        sub = g[g["moran_class"] == cls]
        if len(sub):
            sub.plot(ax=ax, color=col, edgecolor="white", linewidth=0.6, label=f"{cls} ({len(sub)})")
    ax.legend(loc="lower left", fontsize=8)
    _finish(fig, ax, "Local Moran's I — MVI 공간 군집 (p<0.05)",
            f"Queen·행표준화, {C.MORAN_PERMUTATIONS}회 순열, seed {C.MORAN_SEED}", C.MAPS / "local_moran_map.png")


def hotspot_map(dong, freq) -> None:
    g = dong.merge(freq[["dong_code", "hh_share", "hotspot_class"]], on="dong_code")
    fig, ax = _map_axes(g)
    g.plot(ax=ax, column="hh_share", cmap="Reds", vmin=0, vmax=1, edgecolor="white", linewidth=0.6, legend=True,
           legend_kwds={"label": "강건성 격자에서 HH·유의로 나타난 비율", "shrink": 0.6})
    g[g["hotspot_class"] == "stable"].boundary.plot(ax=ax, color="black", linewidth=1.8)
    g[g["hotspot_class"] == "unstable"].boundary.plot(ax=ax, color="black", linewidth=0.8, linestyle="--")
    for _, r in g[g["hotspot_class"] == "stable"].iterrows():
        pt = r.geometry.representative_point()
        ax.annotate(r["dong_name"], (pt.x, pt.y), fontsize=8, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, lw=0))
    n_s, n_u = int((g["hotspot_class"] == "stable").sum()), int((g["hotspot_class"] == "unstable").sum())
    _finish(fig, ax, f"Robust Hotspot — 안정 {n_s}개(굵은 선) / 불안정 {n_u}개(점선)",
            f"안정: 비율 ≥ {C.ROBUST_STABLE_SHARE}, 불안정: ≥ {C.ROBUST_UNSTABLE_SHARE}",
            C.MAPS / "robust_hotspot_map.png")


def policy_map(dong, pol, key_list=("A", "B", "C"), fname="policy_routes_map.png", title=None) -> None:
    comp = pd.read_csv(C.TABLES / "mvi_components.csv", encoding="utf-8-sig", dtype={"dong_code": str})
    g = dong.merge(comp[["dong_code", "mvi"]], on="dong_code")
    fig, ax = _map_axes(g)
    g.plot(ax=ax, column="mvi", cmap="Greys", alpha=0.45, edgecolor="white", linewidth=0.5)
    marker = {"A": "*", "B": "^", "C": "o"}
    size = {"A": 150, "B": 120, "C": 230}  # 같은 정류장을 여러 정책이 고르면 겹치므로 크기·순서를 달리함
    zord = {"A": 7, "B": 6, "C": 5}
    for k in key_list:
        p = pol["policies"][k]
        for r in p["routes"]:
            if r.get("path_lonlat"):
                xy = np.array(r["path_lonlat"])
                ax.plot(xy[:, 0], xy[:, 1], color=POLICY_COLOR[k], linewidth=1.3, alpha=0.8)
        s = pd.DataFrame(p["selected_stops"])
        ax.scatter(s["lon"], s["lat"], c=POLICY_COLOR[k], marker=marker[k], s=size[k], edgecolor="black",
                   zorder=zord[k],
                   label=f"정책 {k}: {p['kpis']['n_new_stops']}개 거점, 취약커버 {p['kpis']['vuln_coverage']:.3f}, "
                         f"{p['kpis']['total_vehicle_km']:.1f}km")
    from src.routing import load_depots
    d = load_depots()
    ax.scatter(d["lon"], d["lat"], c="black", marker="s", s=70, zorder=8, label="차량 거점(구 보건소)")
    ax.legend(loc="lower left", fontsize=7.5)
    _finish(fig, ax, title or "정책 A/B/C — 선정 거점과 차량 순환노선(OSM 차량망 최단경로)",
            f"서비스 반경 {C.COVERAGE_RADIUS_M}m, 속도 {C.VEHICLE_SPEED_KMH}km/h·정차 {C.DWELL_MIN_PER_STOP}분 가정",
            C.MAPS / fname)


def baseline_fig() -> None:
    b = pd.read_csv(C.TABLES / "baseline_comparison.csv", encoding="utf-8-sig")
    b = b[b["method"].isin(["MCLP", "Greedy", "Top_demand", "Random_mean"])].copy()
    b["k"] = b["k"].astype(int)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = {"MCLP": "MCLP(최적)", "Greedy": "탐욕 휴리스틱", "Top_demand": "취약수요 상위 동 우선", "Random_mean": "무작위(20 seeds 평균)"}
    ks = sorted(b["k"].unique())
    width = 0.2
    for i, m in enumerate(labels):
        s = b[b["method"] == m].set_index("k").loc[ks]
        bars = ax.bar(np.arange(len(ks)) + (i - 1.5) * width, s["vuln_coverage"], width, label=labels[m])
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(np.arange(len(ks)), [f"k={k}" for k in ks])
    ax.set_ylabel("취약 고령수요 커버 비율")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    _finish(fig, ax, "입지 방법 비교 — 동일 k·반경·수요정의", "baseline_comparison.csv", C.FIGURES / "baseline_comparison.png")


def coverage_curve_fig() -> None:
    c = pd.read_csv(C.TABLES / "mclp_coverage_curve.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(c["k"], c["vuln_coverage"], marker="o", label="취약수요 커버")
    ax.plot(c["k"], c["elderly_coverage"], marker="s", label="고령인구 커버")
    ax.bar(c["k"], c["marginal_gain"], alpha=0.3, label="한계효과(추가 1개당)")
    ax.set_xlabel("신규 거점 수 k")
    ax.set_ylabel("비율")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    _finish(fig, ax, "MCLP 커버리지 곡선 (k=1~10, 각 k 최적해)", "mclp_coverage_curve.csv", C.FIGURES / "mclp_coverage_curve.png")


def tradeoff_fig(pol, pareto) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    p = pareto[pareto["route_km"].notna()].sort_values("route_km")
    ax.plot(p["route_km"], p["vuln_coverage"], marker="o", color=POLICY_COLOR["C"], alpha=0.7,
            label="정책 C Pareto (예산별 Orienteering 최적)")
    for k in ["A", "B", "C"]:
        kp = pol["policies"][k]["kpis"]
        ax.scatter(kp["total_vehicle_km"], kp["vuln_coverage"], s=160, color=POLICY_COLOR[k], edgecolor="black",
                   zorder=5, label=f"정책 {k}")
        ax.annotate(f"{k}", (kp["total_vehicle_km"], kp["vuln_coverage"]), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("1회 순환 총 운행거리 (vehicle-km)")
    ax.set_ylabel("취약 고령수요 커버 비율")
    ax.legend(fontsize=8)
    _finish(fig, ax, "정책 trade-off — 운행거리 대비 공공효과", "policy_C_pareto.csv, policy_comparison.csv",
            C.FIGURES / "policy_tradeoff.png")


def kpi_table_fig(pol) -> None:
    rows = [("정류장 수", "n_new_stops", "{:.0f}"), ("취약수요 커버", "vuln_coverage", "{:.3f}"),
            (f"상위{C.TOP_K_VULNERABLE} 취약동 커버", f"top{C.TOP_K_VULNERABLE}_coverage", "{:.2f}"),
            ("Hotspot 커버", "hotspot_coverage", "{:.3f}"), ("최소 구 커버", "min_gu_coverage", "{:.3f}"),
            ("구간 격차", "gu_coverage_gap", "{:.3f}"), ("운행거리(km)", "total_vehicle_km", "{:.1f}"),
            ("최대 순환(분)", "max_cycle_min", "{:.1f}"), ("차량(대)", "vehicles_used", "{:.0f}"),
            ("10km당 커버", "vuln_coverage_per_10km", "{:.3f}")]
    cell = [[lab] + [fmt.format(pol["policies"][k]["kpis"][key]) if pol["policies"][k]["kpis"][key] is not None
                     else "-" for k in "ABC"] for lab, key, fmt in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axis("off")
    t = ax.table(cellText=cell, colLabels=["KPI", "A 공공효과", "B 형평성", "C 운영효율"], loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.4)
    _finish(fig, ax, "정책별 동일 KPI 비교", "policy_comparison.csv", C.FIGURES / "policy_kpi_table.png")


def before_after_fig() -> None:
    g = pd.read_csv(C.TABLES / "before_after_by_gu.csv", encoding="utf-8-sig")
    g = g[g["metric"] == "served_elderly"]
    piv = g.pivot(index="gu", columns="policy", values="after")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(piv))
    ax.bar(x - 0.3, np.zeros(len(piv)), 0.2, color="#bbbbbb", label="Before (전담 서비스 없음)")
    for i, k in enumerate(["A", "B", "C"]):
        bars = ax.bar(x + (i - 1) * 0.2 + 0.05, piv[k], 0.2, color=POLICY_COLOR[k], label=f"After — 정책 {k}")
        ax.bar_label(bars, fmt="%.0f", fontsize=7)
    ax.set_xticks(x, piv.index)
    ax.set_ylabel("서비스 반경 내 고령인구(명)")
    ax.legend(fontsize=8)
    _finish(fig, ax, "Before/After — 구별 신규 이동지원 서비스 수혜 고령인구", "before_after_by_gu.csv",
            C.FIGURES / "policy_before_after.png")


def accessibility_comparison_fig() -> None:
    c = pd.read_csv(C.TABLES / "accessibility_method_comparison.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(c["rank_nearest"], c["rank_e2sfca_network"], s=30)
    ax.plot([1, C.N_DONG], [1, C.N_DONG], color="gray", linestyle="--", linewidth=1)
    for _, r in c.head(6).iterrows():
        ax.annotate(r["dong_name"], (r["rank_nearest"], r["rank_e2sfca_network"]), fontsize=8, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("최근접 의료시설 거리 순위 (1=가장 가까움)")
    ax.set_ylabel("E2SFCA 의료 접근성 순위 (1=가장 좋음)")
    s = json.loads((C.JSON_DIR / "accessibility_summary.json").read_text(encoding="utf-8"))["method_comparison"]
    _finish(fig, ax, f"최근접 거리 vs E2SFCA (Spearman ρ={s['spearman_nearest_vs_e2sfca']})",
            "순위 이동이 큰 동 이름 표시", C.FIGURES / "accessibility_method_comparison.png")


def mvi_decomposition_fig(comp) -> None:
    top = comp.nsmallest(C.TOP_K_VULNERABLE, "mvi_rank").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    y = np.arange(len(top))
    ax.barh(y - 0.2, top["need"], 0.4, label="Need (수요)", color="#e6550d")
    ax.barh(y + 0.2, top["access_deficit"], 0.4, label="1 - Access (접근 결핍)", color="#3182bd")
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(1.01, i, f"MVI {r.mvi:.3f} · 최약: {r.weakest_access_label}", va="center", fontsize=8)
    ax.set_yticks(y, [f"{int(r.mvi_rank)}. {r.dong_name}" for _, r in top.iterrows()])
    ax.set_xlim(0, 1.6)
    ax.legend(fontsize=8, loc="lower right")
    _finish(fig, ax, f"MVI 상위 {C.TOP_K_VULNERABLE}개 동 원인 분해", "mvi_components.csv",
            C.FIGURES / "mvi_decomposition_top10.png")


def robustness_fig(freq) -> None:
    f = freq[freq["hh_share"] > 0].sort_values("hh_share")
    colors = f["hotspot_class"].map({"stable": "#b2182b", "unstable": "#ef8a62", "none": "#bbbbbb"})
    fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(f) + 1)))
    ax.barh(f["dong_name"], f["hh_share"], color=colors)
    ax.axvline(C.ROBUST_STABLE_SHARE, color="black", linestyle="--", linewidth=1)
    ax.axvline(C.ROBUST_UNSTABLE_SHARE, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("강건성 격자 시나리오 중 HH·유의 비율")
    ax.set_xlim(0, 1)
    _finish(fig, ax, "Hotspot 출현 빈도 — 안정/불안정 구분", "hotspot_frequency.csv",
            C.FIGURES / "robustness_hotspot_frequency.png")


def interactive_map(dong, comp, freq, pol) -> None:
    import folium
    g = dong.merge(comp[["dong_code", "mvi", "mvi_rank", "need", "access", "weakest_access_label"]], on="dong_code")
    g = g.merge(freq[["dong_code", "hotspot_class", "hh_share"]], on="dong_code")
    x0, y0, x1, y1 = g.total_bounds
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)  # 키 불필요 타일
    m.fit_bounds([[y0, x0], [y1, x1]])
    folium.Choropleth(geo_data=g[["dong_code", "geometry"]].to_json(), data=g, columns=["dong_code", "mvi"],
                      key_on="feature.properties.dong_code", fill_color="YlOrRd", fill_opacity=0.75,
                      legend_name="MVI = Need × (1 - Access)").add_to(m)
    tip = folium.GeoJsonTooltip(fields=["dong_name", "mvi", "mvi_rank", "need", "access", "weakest_access_label",
                                        "hotspot_class"],
                                aliases=["행정동", "MVI", "순위", "Need", "Access", "최약 접근성", "Hotspot"])
    folium.GeoJson(g.drop(columns=[c for c in g.columns if c not in
                                   ["dong_name", "mvi", "mvi_rank", "need", "access", "weakest_access_label",
                                    "hotspot_class", "geometry"]]).to_json(),
                   style_function=lambda f: {"fillOpacity": 0, "weight": 0.5, "color": "#555"}, tooltip=tip).add_to(m)
    for k in "ABC":
        fg = folium.FeatureGroup(name=f"정책 {k}")
        for s in pol["policies"][k]["selected_stops"]:
            folium.CircleMarker([s["lat"], s["lon"]], radius=7, color=POLICY_COLOR[k], fill=True,
                                tooltip=f"정책 {k}: {s['name']} ({s['dong']})").add_to(fg)
        for r in pol["policies"][k]["routes"]:
            if r.get("path_lonlat"):
                folium.PolyLine([[p[1], p[0]] for p in r["path_lonlat"]], color=POLICY_COLOR[k], weight=3,
                                tooltip=f"정책 {k} {r['gu']} 노선 {r['km']}km").add_to(fg)
        fg.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(C.MAPS / "mvi_interactive.html")


def run_visualization() -> None:
    C.MAPS.mkdir(parents=True, exist_ok=True)
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    dong = _dong()
    comp = pd.read_csv(C.TABLES / "mvi_components.csv", encoding="utf-8-sig", dtype={"dong_code": str})
    moran = pd.read_csv(C.TABLES / "local_moran_results.csv", encoding="utf-8-sig", dtype={"dong_code": str})
    freq = pd.read_csv(C.TABLES / "hotspot_frequency.csv", encoding="utf-8-sig", dtype={"dong_code": str})
    pol = json.loads((C.JSON_DIR / "policy_results.json").read_text(encoding="utf-8"))
    pareto = pd.read_csv(C.TABLES / "policy_C_pareto.csv", encoding="utf-8-sig")
    mvi_map(dong, comp)
    moran_map(dong, moran)
    hotspot_map(dong, freq)
    policy_map(dong, pol, ("A",), "mclp_map.png", f"정책 A(MCLP k={C.K_MAIN}) — 선정 거점과 차량 노선")
    policy_map(dong, pol)
    baseline_fig()
    coverage_curve_fig()
    tradeoff_fig(pol, pareto)
    kpi_table_fig(pol)
    before_after_fig()
    accessibility_comparison_fig()
    mvi_decomposition_fig(comp)
    robustness_fig(freq)
    interactive_map(dong, comp, freq, pol)
    print("visualization done")


if __name__ == "__main__":
    run_visualization()
