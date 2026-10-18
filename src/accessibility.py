"""접근성 분석 — 의료·복지·대중교통 3개 구성요소.

수요점 i : 행정동 수요 대표점 50개 (network.demand_points: 동 내 버스정류장 medoid), 수요량 = 65세 이상 인구(명)
거리     : OSM 보행망 네트워크 거리(m, primary) / 직선거리(m, baseline 비교용)

[의료] E2SFCA (Luo & Qi 2009, 거리밴드 감쇠)
  R_j = S_j / Σ_i W(d_ij)·P_i ,  A_i = Σ_j W(d_ij)·R_j
  W: E2SFCA_BANDS (0–1km 1.0 / 1–2km 0.6 / 2–3km 0.25, 3km 초과 0)
  공급량 S1(primary) = 고유 물리시설 1개당 1 (원자료에 병상·의사수 없음 → facility-count proxy)
  공급량 S2(민감도)  = 종별 가중 (종합병원 5, 병원 3, 요양·치과·한방병원 2, 의원급 1)
  보조지표: 최근접 시설 거리, 1km 내 시설 수 (nearest-distance baseline)
[복지] E2SFCA, 공급량 = 재가장기요양기관 중 시설이용형(주야간보호·단기보호) 정원(명).
  방문형 급여(방문요양·목욕·간호)는 제공자가 가정을 방문하므로 거리 기반 접근성에 쓰지 않는다.
[대중교통] 고령인구 1,000명당 버스정류장 수 (동 내 정류장 수 기준) + 최근접 정류장 보행거리(참고)

경계 처리: 성남시 폴리곤 안으로 지오코딩된 시설만 공급에 포함(시 경계 밖 시설 미포함 → 경계부
접근성 과소추정 가능, docs/limitations.md). 캐치먼트 안에 수요가 없는 시설은 R_j=0.
성질: Σ_i P_i·A_i = Σ_{j: 캐치먼트 내 수요 존재} S_j (공급 보존 — tests에서 검증).

출력: data/processed/accessibility.csv, outputs/tables/accessibility_by_dong.csv,
      outputs/tables/accessibility_method_comparison.csv, outputs/json/accessibility_summary.json
      data/interim/distance_matrices.npz (민감도·What-if 재사용용 캐시)
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from scipy.stats import spearmanr

from src import config as C
from src import network as N


def s2_weight(facility_types: str) -> float:
    """종별 가중치 — '병원' 부분문자열 함정(요양병원 등) 방지를 위해 우선순위 매칭."""
    t = str(facility_types or "")
    if "종합병원" in t:
        return 5.0
    if "요양병원" in t:
        return 2.0
    if "치과병원" in t or "한방병원" in t:
        return 2.0
    if "병원" in t:  # 일반 병원·정신병원 (위 항목에 걸리지 않은 경우)
        return 3.0
    return 1.0


def band_weight(dist_m: float, bands=None) -> float:
    """단일 거리(m)의 E2SFCA 밴드 가중치."""
    for lo, hi, w in (bands or C.E2SFCA_BANDS):
        if lo <= dist_m < hi:
            return w
    return 0.0


def band_weights(D: np.ndarray, bands=None) -> np.ndarray:
    """거리행렬(m) → 밴드 가중치 행렬 (벡터화). inf(도달불가)는 0."""
    W = np.zeros_like(D, dtype=float)
    for lo, hi, w in (bands or C.E2SFCA_BANDS):
        W[(D >= lo) & (D < hi)] = w
    return W


def e2sfca(D: np.ndarray, demand: np.ndarray, supply: np.ndarray, bands=None) -> tuple[np.ndarray, np.ndarray]:
    """E2SFCA. D: 수요점×시설 거리(m), demand: 수요점 인구(명), supply: 시설 공급량.

    반환 (A_i: 수요점별 접근성 [공급단위/명], R_j: 시설별 공급/수요비).
    캐치먼트 안 수요가 0인 시설은 R_j=0 (0으로 나누기 방지).
    """
    if D.shape != (len(demand), len(supply)):
        raise ValueError(f"차원 불일치: D{D.shape}, demand{len(demand)}, supply{len(supply)}")
    if (demand < 0).any() or (supply < 0).any():
        raise ValueError("수요·공급은 음수일 수 없음")
    W = band_weights(D, bands)
    denom = (W * demand[:, None]).sum(axis=0)
    R = np.divide(supply, denom, out=np.zeros_like(supply, dtype=float), where=denom > 0)
    A = (W * R[None, :]).sum(axis=1)
    return A, R


@dataclass
class AccessInputs:
    dong: gpd.GeoDataFrame           # DONG_MASTER 순서 (분석 전체의 기준 순서)
    demand: np.ndarray               # 65세 이상 인구
    cent_lon: np.ndarray
    cent_lat: np.ndarray
    medical: pd.DataFrame            # 성남시 폴리곤 내 지오코딩 성공 의료시설
    welfare: pd.DataFrame            # 성남시 폴리곤 내 지오코딩 성공 시설이용형 재가요양기관
    bus: pd.DataFrame                # 행정동 매칭 버스정류장
    features: pd.DataFrame


def load_inputs() -> AccessInputs:
    dong = gpd.read_file(C.DONG_MASTER, encoding="utf-8")
    dong["dong_code"] = dong["dong_code"].astype(str)
    feat = pd.read_csv(C.DONG_FEATURES, encoding="utf-8-sig", dtype={"dong_code": str})
    feat = feat.set_index("dong_code").loc[dong["dong_code"]].reset_index()
    dp = N.demand_points(dong)
    dp.to_csv(C.PROCESSED_DIR / "demand_points.csv", index=False, encoding="utf-8-sig")
    lon, lat = dp["lon"].to_numpy(), dp["lat"].to_numpy()

    if not C.MEDICAL_GEOCODED.exists():
        raise FileNotFoundError(f"{C.MEDICAL_GEOCODED} 없음 — python -m src.geocoding 실행 필요")
    geo = pd.read_csv(C.MEDICAL_GEOCODED, encoding="utf-8-sig")
    med = geo[(geo["geocode_status"] == "success") & (geo["in_seongnam"].astype(str) == "True")].copy()
    if len(med) == 0:
        raise RuntimeError("지오코딩 성공 의료시설 0건 — accessibility 계산 불가")
    clean = pd.read_csv(C.MEDICAL_CLEAN, encoding="utf-8-sig").drop_duplicates("unique_facility_id")
    med["facility_types"] = med["unique_facility_id"].map(
        clean.set_index("unique_facility_id")["facility_types"])
    if med["facility_types"].isna().any():
        raise RuntimeError("지오코딩 시설 중 medical_facilities_clean에 없는 ID 존재")
    med["supply_S1"] = 1.0
    med["supply_S2"] = med["facility_types"].map(s2_weight)

    if not C.CARE_GEOCODED.exists():
        raise FileNotFoundError(f"{C.CARE_GEOCODED} 없음 — python -m src.geocoding care 실행 필요")
    cg = pd.read_csv(C.CARE_GEOCODED, encoding="utf-8-sig")
    care = pd.read_csv(C.CARE_CLEAN, encoding="utf-8-sig")
    wel = care.merge(cg[["care_id", "latitude", "longitude", "geocode_status", "in_seongnam"]],
                     on="care_id", how="left", validate="one_to_one")
    wel = wel[wel["facility_based"] & (wel["geocode_status"] == "success")
              & (wel["in_seongnam"].astype(str) == "True")].copy()
    if len(wel) == 0:
        raise RuntimeError("지오코딩된 시설이용형 재가요양기관 0건 — 복지 접근성 계산 불가")

    bus = pd.read_csv(C.BUS_CLEAN, encoding="utf-8-sig")
    bus = bus[bus["join_status"] == "matched"].copy()
    return AccessInputs(dong, feat["elderly_population"].to_numpy(float), lon, lat,
                        med.reset_index(drop=True), wel.reset_index(drop=True),
                        bus.reset_index(drop=True), feat)


def _euclid(lon1, lat1, lon2, lat2) -> np.ndarray:
    tr = Transformer.from_crs(C.CRS_WGS84, C.CRS_METRIC, always_xy=True)
    a = np.column_stack(tr.transform(lon1, lat1))
    b = np.column_stack(tr.transform(lon2, lat2))
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))


def distance_matrices(inp: AccessInputs) -> dict[str, np.ndarray]:
    """수요점×시설 거리행렬. 네트워크 거리는 민감도 최대 캐치먼트까지만 탐색."""
    limit = max(hi for _, hi, _ in C.E2SFCA_BANDS) * max(C.DECAY_SCALE_VARIANTS)
    walk = N.load("walk")
    mats = {
        "med_euclid": _euclid(inp.cent_lon, inp.cent_lat,
                              inp.medical["longitude"].to_numpy(float), inp.medical["latitude"].to_numpy(float)),
        "med_network": walk.distance_matrix(inp.cent_lon, inp.cent_lat, inp.medical["longitude"].to_numpy(float),
                                            inp.medical["latitude"].to_numpy(float), limit=limit),
        "wel_network": walk.distance_matrix(inp.cent_lon, inp.cent_lat, inp.welfare["longitude"].to_numpy(float),
                                            inp.welfare["latitude"].to_numpy(float), limit=limit),
        "bus_network": walk.distance_matrix(inp.cent_lon, inp.cent_lat, inp.bus["경도"].to_numpy(float),
                                            inp.bus["위도"].to_numpy(float), limit=limit),
    }
    return mats


def scaled_bands(scale: float) -> list[tuple[float, float, float]]:
    """거리감쇠 민감도: 밴드 경계를 scale배 (가중치는 동일)."""
    return [(lo * scale, hi * scale, w) for lo, hi, w in C.E2SFCA_BANDS]


def compute_access_table(inp: AccessInputs, mats: dict[str, np.ndarray], bands=None) -> pd.DataFrame:
    """동별 접근성 지표 표. bands를 바꿔 민감도 분석에 재사용한다."""
    bands = bands or C.E2SFCA_BANDS
    P = inp.demand
    med_net, _ = e2sfca(mats["med_network"], P, inp.medical["supply_S1"].to_numpy(), bands)
    med_s2, _ = e2sfca(mats["med_network"], P, inp.medical["supply_S2"].to_numpy(), bands)
    med_euc, _ = e2sfca(mats["med_euclid"], P, inp.medical["supply_S1"].to_numpy(), bands)
    wel_net, _ = e2sfca(mats["wel_network"], P, inp.welfare["capacity"].to_numpy(float), bands)
    catch = max(hi for _, hi, _ in bands)
    out = pd.DataFrame({
        "dong_code": inp.dong["dong_code"].values,
        "dong_name": inp.dong["dong_name"].values,
        "elderly_population": P.astype(int),
        # 의료 — nearest-distance baseline
        "med_nearest_network_m": np.round(mats["med_network"].min(axis=1), 1),
        "med_nearest_euclid_m": np.round(mats["med_euclid"].min(axis=1), 1),
        "med_within_1km_network": (mats["med_network"] < C.NEAR_THRESHOLD_M).sum(axis=1),
        # 의료 — E2SFCA (시설 수/고령인구 1,000명)
        "med_e2sfca_network": med_net * 1000,
        "med_e2sfca_euclid": med_euc * 1000,
        "med_e2sfca_S2_network": med_s2 * 1000,
        # 복지 — E2SFCA (시설이용형 정원 석/고령인구 1,000명)
        "wel_e2sfca_network": wel_net * 1000,
        "wel_nearest_network_m": np.round(mats["wel_network"].min(axis=1), 1),
        "wel_capacity_in_catchment": (np.where(mats["wel_network"] < catch, 1, 0)
                                      * inp.welfare["capacity"].to_numpy(float)[None, :]).sum(axis=1),
        # 대중교통
        "bus_stop_count": inp.features["bus_stop_count"].to_numpy(int),
        "transit_stops_per_1000_elderly": inp.features["bus_stop_count"].to_numpy(float) / P * 1000,
        "bus_nearest_network_m": np.round(mats["bus_network"].min(axis=1), 1),
    })
    return out


def _method_comparison(acc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """최근접거리 vs E2SFCA, 직선 vs 네트워크 E2SFCA 순위 비교 (높을수록 접근성 좋음으로 통일)."""
    fin = acc["med_nearest_network_m"].replace(np.inf, np.nan)
    r_near = (-fin).rank(ascending=False, method="min", na_option="bottom")
    r_e2 = acc["med_e2sfca_network"].rank(ascending=False, method="min")
    r_euc = acc["med_e2sfca_euclid"].rank(ascending=False, method="min")
    comp = acc[["dong_code", "dong_name", "elderly_population", "med_nearest_network_m",
                "med_within_1km_network", "med_e2sfca_network", "med_e2sfca_euclid"]].copy()
    comp["rank_nearest"] = r_near.astype(int)
    comp["rank_e2sfca_network"] = r_e2.astype(int)
    comp["rank_e2sfca_euclid"] = r_euc.astype(int)
    comp["rank_shift_nearest_to_e2sfca"] = comp["rank_e2sfca_network"] - comp["rank_nearest"]
    comp = comp.sort_values("rank_shift_nearest_to_e2sfca", key=np.abs, ascending=False)
    rho_ne, _ = spearmanr(r_near, r_e2)
    rho_en, _ = spearmanr(r_e2, r_euc)
    worst15_near = set(acc.index[r_near > len(acc) - 15])
    worst15_e2 = set(acc.index[r_e2 > len(acc) - 15])
    summary = {
        "spearman_nearest_vs_e2sfca": round(float(rho_ne), 4),
        "spearman_e2sfca_network_vs_euclid": round(float(rho_en), 4),
        "bottom15_overlap_nearest_vs_e2sfca": len(worst15_near & worst15_e2),
        "largest_rank_shifts": comp.head(5)[["dong_name", "rank_nearest", "rank_e2sfca_network",
                                             "med_nearest_network_m", "med_within_1km_network",
                                             "elderly_population"]].to_dict("records"),
        "interpretation": ("최근접 거리는 '가장 가까운 시설 1곳'만 보고, E2SFCA는 캐치먼트 내 시설 수와 "
                           "그 시설을 함께 이용하는 고령인구(경쟁)를 반영한다. 순위가 크게 이동한 동은 "
                           "가까운 시설은 있으나 주변 수요가 많거나(하락), 최근접 시설은 멀지만 캐치먼트 내 "
                           "시설이 많은(상승) 경우이다."),
    }
    return comp, summary


def run_accessibility() -> pd.DataFrame:
    inp = load_inputs()
    mats = distance_matrices(inp)
    C.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.INTERIM_DIR / "distance_matrices.npz", **mats)
    acc = compute_access_table(inp, mats)
    fin_cols = ["med_e2sfca_network", "med_e2sfca_euclid", "med_e2sfca_S2_network",
                "wel_e2sfca_network", "transit_stops_per_1000_elderly"]
    if not np.isfinite(acc[fin_cols].to_numpy()).all():
        raise RuntimeError("접근성 지표에 NaN/inf 존재")
    if (acc[fin_cols] < 0).any().any():
        raise RuntimeError("접근성 지표에 음수 존재")
    acc.to_csv(C.PROCESSED_DIR / "accessibility.csv", index=False, encoding="utf-8-sig")
    acc.to_csv(C.TABLES / "accessibility_by_dong.csv", index=False, encoding="utf-8-sig")
    comp, comp_summary = _method_comparison(acc)
    comp.to_csv(C.TABLES / "accessibility_method_comparison.csv", index=False, encoding="utf-8-sig")

    walk = N.load("walk")
    # 수요점 민감도: 폴리곤 중심점을 수요점으로 썼을 때의 의료 E2SFCA와 비교
    dp = pd.read_csv(C.PROCESSED_DIR / "demand_points.csv", encoding="utf-8-sig", dtype={"dong_code": str})
    limit = max(hi for _, hi, _ in C.E2SFCA_BANDS) * max(C.DECAY_SCALE_VARIANTS)
    Dc = walk.distance_matrix(dp["centroid_lon"].to_numpy(), dp["centroid_lat"].to_numpy(),
                              inp.medical["longitude"].to_numpy(float), inp.medical["latitude"].to_numpy(float),
                              limit=limit)
    med_c, _ = e2sfca(Dc, inp.demand, inp.medical["supply_S1"].to_numpy())
    _, cs = walk.snap(dp["centroid_lon"].to_numpy(), dp["centroid_lat"].to_numpy())
    demand_point_note = {
        "method": "동 내 버스정류장 medoid (정류장 없으면 폴리곤 내부 대표점)",
        "n_bus_stop_medoid": int((dp["method"] == "bus_stop_medoid").sum()),
        "shift_from_centroid_m": {"median": float(dp["shift_m"].median()), "max": float(dp["shift_m"].max())},
        "spearman_med_e2sfca_vs_centroid_demand": round(float(spearmanr(acc["med_e2sfca_network"], med_c * 1000)[0]), 4),
        "dongs_zero_medical_with_centroid": acc.loc[med_c == 0, "dong_name"].tolist(),
        "dongs_zero_medical_with_medoid": acc.loc[acc["med_e2sfca_network"] == 0, "dong_name"].tolist(),
        "centroid_snap_max_m": round(float(cs.max()), 1),
        "reason": "산지·공원을 포함한 동은 폴리곤 중심점이 거주지와 떨어져 네트워크 거리가 과대추정됨(예: 중심점 기준 구미1동 → 동 내 의료시설 3km+)",
    }
    _, csnap = walk.snap(inp.cent_lon, inp.cent_lat)
    _, msnap = walk.snap(inp.medical["longitude"].to_numpy(float), inp.medical["latitude"].to_numpy(float))
    reach = np.isfinite(mats["med_network"]).any(axis=0)
    summary = {
        "parameters": {"e2sfca_bands_m": C.E2SFCA_BANDS, "near_threshold_m": C.NEAR_THRESHOLD_M,
                       "distance": "OSM 보행망 네트워크 거리(primary) / 직선거리(baseline)",
                       "demand": "행정동 65세 이상 인구",
                       "demand_point": "행정동 수요 대표점(동 내 버스정류장 medoid, 정류장 없으면 폴리곤 내부 대표점)"},
        "supply": {
            "medical_S1": "고유 물리시설 1개 = 1 (원자료에 병상·의사수 없음, facility-count proxy)",
            "medical_S2": "종별 가중(민감도): 종합병원5/병원3/요양·치과·한방병원2/의원급1",
            "welfare": "시설이용형(주야간보호·단기보호) 정원(명)",
            "n_medical_used": int(len(inp.medical)), "n_welfare_used": int(len(inp.welfare)),
            "welfare_capacity_total": int(inp.welfare["capacity"].sum()),
            "n_medical_reachable_by_any_dong": int(reach.sum()),
        },
        "snap_distance_m": {"demand_point_median": round(float(np.median(csnap)), 1),
                            "demand_point_max": round(float(csnap.max()), 1),
                            "medical_median": round(float(np.median(msnap)), 1),
                            "medical_max": round(float(msnap.max()), 1)},
        "method_comparison": comp_summary,
        "demand_point": demand_point_note,
    }
    (C.JSON_DIR / "accessibility_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (C.TABLES / "supply_definition_note.txt").write_text(
        "의료 S1: 원자료에 병상/의사수 등 실제 공급량 컬럼 없음 → facility-count proxy. "
        "S2: 종별 가중(민감도 전용). 복지: 시설이용형 재가요양기관 정원(명). "
        "방문형 급여는 거리 기반 접근성 미적용.", encoding="utf-8")
    print(f"accessibility: dongs={len(acc)}, medical={len(inp.medical)}, welfare={len(inp.welfare)}, "
          f"rho(nearest,e2sfca)={comp_summary['spearman_nearest_vs_e2sfca']}, "
          f"rho(network,euclid)={comp_summary['spearman_e2sfca_network_vs_euclid']}")
    return acc


if __name__ == "__main__":
    run_accessibility()
