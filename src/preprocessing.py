"""전처리 — data/raw 원본 → data/processed 가공 데이터 (파이프라인 Stage 0).

원본은 읽기만 하고 수정하지 않는다. 삭제·통합·매칭 실패 레코드는 버리지 않고
outputs/unmatched_records.csv, outputs/boundary_exceptions.csv에 사유와 함께 기록한다.

산출물:
  data/processed/population_clean.csv      고령인구 (동 50개, 전체 월 보존)
  data/processed/bus_stops_clean.csv       버스정류장 + 행정동 spatial join
  data/processed/medical_facilities_clean.csv  의료기관 (텍스트 정규화 + 고유시설 ID)
  data/processed/seongnam_dong_master.geojson  행정동 경계 + 동 단위 특성
  data/processed/dong_features_base.csv    동 단위 특성 (분석 입력)
  outputs/geocoding_targets_medical.csv    지오코딩 대상 (고유시설 기준)
  outputs/preprocessing_report.md          단계별 수치 리포트
의료기관 좌표는 원본에 없으므로 여기서 매칭하지 않는다(src/geocoding.py 담당, 임의 좌표 금지).
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points

from src import config as C

CODE_IN_NAME = re.compile(r"^(?P<name>.*?)\s*\((?P<code>\d{10})\)\s*$")
BOUNDARY_ROUNDING_M = 30.0  # 경계 밖 정류장이 이 거리 이내면 좌표 반올림 오차로 판단


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371000.0 * math.asin(math.sqrt(a))


def _norm_text(v):
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v).strip()
    return v


def load_boundary() -> gpd.GeoDataFrame:
    b = gpd.read_file(C.RAW_BOUNDARY, encoding="utf-8")
    assert len(b) == C.N_DONG, f"행정동 경계 {len(b)}개 (기대 {C.N_DONG})"
    b["dong_code"] = b["adm_cd2"].astype(str)
    b["dong_name"] = b["adm_nm"].str.split().str[-1]
    b["adm_nm_full"] = b["adm_nm"]
    return b


def clean_population(unmatched: list[dict], report: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(C.RAW_POPULATION, encoding="utf-8-sig", dtype=str)
    report.append(f"- 원본 행 수: {len(raw)}")
    parsed = raw["행정구역"].str.extract(CODE_IN_NAME)
    assert parsed["code"].notna().all(), "행정구역 코드 파싱 실패 행 존재"
    # 시·구 집계행: 코드 끝 5자리가 00000 (예: 4113000000 성남시, 4113100000 수정구)
    is_agg = parsed["code"].str[-5:] == "00000"
    for ident in raw.loc[is_agg, "행정구역"]:
        unmatched.append({"source_dataset": "고령인구현황", "identifier": ident,
                          "issue_type": "aggregate_row_removed",
                          "reason": "시/구 단위 집계행 (동 단위 마스터 대상 아님)"})
    agg_names = parsed.loc[is_agg, "name"].str.strip().tolist()
    report.append(f"- 시/구 단위 집계행으로 판별되어 제거된 행: {int(is_agg.sum())}건 -> {agg_names}")

    df = raw.loc[~is_agg].copy()
    month_cols = [c for c in raw.columns if c != "행정구역"]
    months = sorted({c.split("_")[0] for c in month_cols})
    assert months[-1] == C.BASE_MONTH, f"기준월 {C.BASE_MONTH}이 데이터 최신월({months[-1]})과 다름"
    for c in month_cols:
        df[c] = df[c].str.replace(",", "", regex=False).astype(int)
    out = pd.DataFrame({
        "dong_code": parsed.loc[~is_agg, "code"].values,
        "dong_name": parsed.loc[~is_agg, "name"].str.split().str[-1].values,
        "region_full_name": parsed.loc[~is_agg, "name"].str.strip().values,
        "total_population": df[f"{C.BASE_MONTH}_전체"].values,
        "elderly_population": df[f"{C.BASE_MONTH}_65세이상전체"].values,
    })
    out = pd.concat([out, df[month_cols].reset_index(drop=True)], axis=1)
    assert out["dong_code"].is_unique, "dong_code 중복"
    report.append(f"- 동 단위 잔존 행 수: {len(out)}")
    report.append(f"- 감지된 월 목록: {months[0]} ~ {months[-1]} (총 {len(months)}개월)")
    report.append(f"- 마스터 데이터셋에 사용할 기준월: **{C.BASE_MONTH}** (데이터 내 최신월, 임의 추정 아님)")
    report.append(f"- dong_code 중복 건수: {int(out['dong_code'].duplicated().sum())}")
    out.to_csv(C.POPULATION_CLEAN, index=False, encoding="utf-8-sig")
    report.append(f"- 저장: data/processed/population_clean.csv ({len(out)}행, {out.shape[1]}열)")
    return out


def clean_bus(boundary: gpd.GeoDataFrame, unmatched: list[dict],
              report: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(C.RAW_BUS, encoding="utf-8-sig")
    report.append(f"- 원본 행 수: {len(raw)}")
    df = raw.dropna(how="all").copy()
    report.append(f"- 완전 공백 행 제거: {len(raw) - len(df)}건")
    report.append(f"- 잔존 행 수: {len(df)}")
    df = df.rename(columns={"행정동": "raw_dong_label"})
    for c in ["위도", "경도"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
    df["정류장번호(ID)"] = pd.to_numeric(df["정류장번호(ID)"], errors="coerce")

    has_xy = df["위도"].notna() & df["경도"].notna()
    for _, r in df[~has_xy].iterrows():
        ident = r["정류장번호(ID)"]
        unmatched.append({"source_dataset": "버스정류장",
                          "identifier": "" if pd.isna(ident) else str(ident),
                          "issue_type": "missing_coordinates",
                          "reason": "위도/경도 결측으로 spatial join 불가 (원본 그대로 보존)"})
    report.append(f"- 좌표 결측 행(완전공백은 아니나 유효 데이터 없음): {int((~has_xy).sum())}건 "
                  "(삭제하지 않고 unmatched_records.csv에 기록)")

    pts = gpd.GeoDataFrame(df[has_xy].copy(), geometry=gpd.points_from_xy(
        df.loc[has_xy, "경도"], df.loc[has_xy, "위도"]), crs=C.CRS_WGS84)
    joined = gpd.sjoin(pts, boundary[["dong_code", "dong_name", "geometry"]].to_crs(C.CRS_WGS84),
                       how="left", predicate="within")
    assert joined.index.is_unique, "정류장이 복수 행정동에 매칭됨 (경계 중첩)"
    df["official_dong_code"] = pd.Series(dtype=object)
    df["official_dong_name"] = pd.Series(dtype=object)
    df.loc[joined.index, "official_dong_code"] = joined["dong_code"]
    df.loc[joined.index, "official_dong_name"] = joined["dong_name"]
    df["join_status"] = np.where(~has_xy, "no_coordinates",
                                 np.where(df["official_dong_code"].notna(), "matched", "outside_polygon"))
    n_m = int((df["join_status"] == "matched").sum())
    n_o = int((df["join_status"] == "outside_polygon").sum())
    report.append(f"- spatial join 결과: matched={n_m}, outside_polygon={n_o}, "
                  f"no_coordinates={int((~has_xy).sum())}")
    report.append(f"- spatial join 성공률(전체 {len(df)}건 기준): {n_m / len(df) * 100:.1f}%")
    report.append(f"- spatial join 성공률(좌표 보유 {int(has_xy.sum())}건 기준): "
                  f"{n_m / int(has_xy.sum()) * 100:.1f}%")
    m = df[df["join_status"] == "matched"]
    same = int((m["raw_dong_label"] == m["official_dong_name"]).sum())
    report.append(f"- 원본 raw_dong_label vs official_dong_name 비교(참고용, 신뢰 안 함): "
                  f"일치 {same}건 / 불일치 {len(m) - same}건")

    # 경계 밖 정류장: 가장 가까운 행정동과 경계까지 거리 기록 (삭제·강제배정 안 함)
    # 거리 = 경위도 좌표계에서 찾은 폴리곤 최근접점까지의 haversine 거리(구면 R=6371km).
    # (최초 전처리와 동일한 방식 — 투영좌표 EPSG:5186 거리와의 차이는 0.03m 이내)
    exc_rows = []
    bm = boundary.to_crs(C.CRS_WGS84).reset_index(drop=True)
    outside = df[df["join_status"] == "outside_polygon"]
    for _, r in outside.iterrows():
        p = Point(r["경도"], r["위도"])
        d = [_haversine_m(p.x, p.y, q.x, q.y)
             for q in (nearest_points(g, p)[0] for g in bm.geometry)]
        i = int(np.argmin(d))
        dist = round(float(d[i]), 2)
        exc_rows.append({"정류장번호(ID)": r["정류장번호(ID)"], "정류장명": r["정류장명"],
                         "raw_dong_label": r["raw_dong_label"], "위도": r["위도"], "경도": r["경도"],
                         "nearest_dong_code": bm.iloc[i]["dong_code"],
                         "nearest_dong_name": bm.iloc[i]["dong_name"],
                         "distance_to_nearest_boundary_m": dist,
                         "likely_boundary_rounding_error": dist <= BOUNDARY_ROUNDING_M})
        report.append(f"  - 경계 밖 정류장 '{r['정류장명']}' → 가장 가까운 행정동 "
                      f"'{bm.iloc[i]['dong_name']}'까지 {dist:.1f}m "
                      f"({BOUNDARY_ROUNDING_M:.0f}m 이하={dist <= BOUNDARY_ROUNDING_M})")
    exc = pd.DataFrame(exc_rows, columns=["정류장번호(ID)", "정류장명", "raw_dong_label", "위도", "경도",
                                          "nearest_dong_code", "nearest_dong_name",
                                          "distance_to_nearest_boundary_m",
                                          "likely_boundary_rounding_error"])
    df.to_csv(C.BUS_CLEAN, index=False, encoding="utf-8-sig")
    exc.to_csv(C.OUTPUTS / "boundary_exceptions.csv", index=False, encoding="utf-8-sig")
    report.append(f"- 저장: data/processed/bus_stops_clean.csv ({len(df)}행) — "
                  "경계 밖/좌표결측 레코드도 삭제하지 않고 포함")
    report.append(f"- 좌표 범위: 위도 {df['위도'].min()} ~ {df['위도'].max()}, "
                  f"경도 {df['경도'].min()} ~ {df['경도'].max()} (CRS: WGS84 / EPSG:4326)")
    return df, exc


def clean_medical(unmatched: list[dict], report: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(C.RAW_MEDICAL, encoding="utf-8-sig", dtype=str)
    df = raw.apply(lambda col: col.map(_norm_text))
    report.append(f"- 원본/정제 후 행 수: {len(df)} (텍스트 정규화만 수행, 행 삭제 없음)")
    has_xy = {"위도", "경도"} <= set(df.columns)
    report.append(f"- 좌표(위도/경도) 컬럼 존재 여부: {has_xy}")
    df["needs_geocoding"] = not has_xy

    # 물리적 고유시설 = (기관명 + 주소). 동일 건물의 종별 복수 등록은 1개 시설로 통합
    key = df["의료기관명"] + "||" + df["의료기관주소(도로명)"]
    first_order = pd.unique(key)
    id_map = {k: f"MED_{i + 1:05d}" for i, k in enumerate(first_order)}
    df["unique_facility_id"] = key.map(id_map)
    types = df.groupby("unique_facility_id", sort=False)["의료기관종별"].agg(
        lambda s: ";".join(sorted(set(s))))
    df["facility_types"] = df["unique_facility_id"].map(types)
    df["official_dong_code"] = pd.Series(dtype=object)
    df["official_dong_name"] = pd.Series(dtype=object)
    df["join_status"] = "no_coordinates_needs_geocoding"

    dup = df[key.duplicated(keep=False)]
    report.append(f"- (기관명+주소) 동일 레코드: {len(dup)}건 → 원본 행은 삭제하지 않고 모두 보존")
    for _, r in dup.iterrows():
        n = int((df["unique_facility_id"] == r["unique_facility_id"]).sum())
        unmatched.append({"source_dataset": "의료기관", "identifier": r["unique_facility_id"],
                          "issue_type": "duplicate_name_address_preserved",
                          "reason": f"동일 (기관명+주소) {n}건 → unique_facility_id로 통합, "
                                    f"원본행은 보존 (종별: {r['facility_types']})"})
    for uid in df["unique_facility_id"]:
        unmatched.append({"source_dataset": "의료기관", "identifier": uid,
                          "issue_type": "missing_coordinates",
                          "reason": "위도/경도 컬럼 없음 → geocoding 필요, spatial join 미수행(임의 매핑 금지)"})
    n_u = df["unique_facility_id"].nunique()
    report.append(f"- 원본 레코드 {len(df)}건 -> 물리적 고유시설(unique_facility_id) {n_u}개 "
                  f"(중복 {len(df) - n_u}건 통합 대상)")
    report.append("- 의료기관 spatial join 성공률: 0.0% (좌표 0건 보유 → geocoding 선행 필요)")

    targets = df.drop_duplicates("unique_facility_id")[
        ["unique_facility_id", "의료기관명", "의료기관주소(도로명)", "구별"]]
    df.to_csv(C.MEDICAL_CLEAN, index=False, encoding="utf-8-sig")
    targets.to_csv(C.GEOCODE_TARGETS, index=False, encoding="utf-8-sig")
    report.append(f"- geocoding 대상 목록: 전체 {len(df)}건 (unique_facility_id 기준 {n_u}개 고유시설)")
    report.append(f"- 저장: outputs/geocoding_targets_medical.csv ({len(targets)}건, 고유시설 기준)")
    return df


def clean_care(report: list[str]) -> pd.DataFrame:
    """재가장기요양기관 정제.

    급여종류 컬럼은 '○'(제공) 또는 공란으로 구성된다. 공란은 원자료상 '표기 없음'이며
    여기서는 '해당 급여 미표기(False)'로만 기록하고 다른 값으로 보간하지 않는다.
    시설 이용형 급여(주야간보호·단기보호)는 정원(명)이 기재되어 있어 복지 접근성의
    공급량으로 사용한다(src/accessibility.py). 방문형 급여(방문요양·목욕·간호)는
    제공자가 이용자 가정을 방문하므로 '기관까지의 거리' 기반 접근성 지표로 쓰지 않는다.
    """
    raw = pd.read_csv(C.RAW_CARE, encoding="utf-8-sig", dtype=str)
    df = raw.dropna(how="all").apply(lambda col: col.map(_norm_text)).reset_index(drop=True)
    report.append(f"- 원본 행 수: {len(raw)} (완전 공백행 {len(raw) - len(df)}건 포함) → 유효 데이터 {len(df)}행")
    svc_cols = [c for c in df.columns if c.startswith("급여종류_")]
    for c in svc_cols:
        n = int(df[c].isna().sum())
        report.append(f"  - `{c}` 미표기(공란, 보간하지 않음): {n}/{len(df)} ({n / len(df) * 100:.1f}%)")
    out = pd.DataFrame({
        "care_id": [f"CARE_{i + 1:04d}" for i in range(len(df))],
        "facility_name": df["기관명"],
        "raw_address": df["소재지"],
        "gu": df["소재지"].str.extract(r"성남시\s+(\S+구)")[0],
    })
    for c in svc_cols:
        out[c.replace("급여종류_", "svc_")] = df[c].eq("○")
    out["facility_based"] = out["svc_주야간보호"] | out["svc_단기보호"]
    out["capacity"] = pd.to_numeric(df["주야간보호_단기보호정원"], errors="coerce")
    out["reference_date"] = df["데이터기준일"]
    fb = out[out["facility_based"]]
    assert fb["capacity"].notna().all(), "시설이용형 급여 기관 중 정원 결측 존재 — 공급량 정의 불가"
    assert out.loc[~out["facility_based"], "capacity"].isna().all(), "방문형 전용 기관에 정원 기재됨 — 원자료 확인 필요"
    report.append(f"- 시설이용형(주야간보호·단기보호) {len(fb)}개소, 정원 합계 {int(fb['capacity'].sum())}명 "
                  "(복지 접근성 E2SFCA 공급량)")
    report.append(f"- 방문형 전용 {int((~out['facility_based']).sum())}개소 — 거리 기반 접근성 미적용 (방문 서비스)")
    out.to_csv(C.CARE_CLEAN, index=False, encoding="utf-8-sig")
    out[["care_id", "facility_name", "raw_address", "gu"]].to_csv(
        C.GEOCODE_TARGETS_CARE, index=False, encoding="utf-8-sig")
    report.append(f"- 저장: data/processed/care_facilities_clean.csv ({len(out)}행), "
                  f"outputs/geocoding_targets_care.csv")
    return out


def build_dong_features(boundary: gpd.GeoDataFrame, pop: pd.DataFrame, bus: pd.DataFrame,
                        report: list[str]) -> gpd.GeoDataFrame:
    area = pd.read_csv(C.RAW_AREA, encoding="utf-8-sig", dtype=str)
    area["gu"] = area["구청명"].str.strip()
    area["dong_name"] = area["동 명"].str.strip()
    area["area_km2"] = area["면적(제곱킬로미터)"].astype(float)
    p = pop[["dong_code", "dong_name", "region_full_name", "total_population", "elderly_population"]].copy()
    p["gu"] = p["region_full_name"].str.split().str[-2]
    p = p.merge(area[["gu", "dong_name", "area_km2"]], on=["gu", "dong_name"],
                how="left", validate="one_to_one")
    assert p["area_km2"].notna().all(), \
        f"면적 매칭 실패: {p.loc[p['area_km2'].isna(), 'dong_name'].tolist()}"

    matched = set(pop["dong_code"]) & set(boundary["dong_code"])
    report.append(f"- population dong_code ↔ GeoJSON adm_cd2 매칭: {len(matched)}/{C.N_DONG} "
                  f"({len(matched) / C.N_DONG * 100:.1f}%)")
    assert len(matched) == C.N_DONG, "인구-경계 코드 매칭 100% 아님"
    report.append(f"- 행정구역 면적 매칭(구+동명): {int(p['area_km2'].notna().sum())}/{C.N_DONG}")

    stops = bus[bus["join_status"] == "matched"].groupby("official_dong_code").size()
    m = boundary[["dong_code", "dong_name", "adm_nm_full", "geometry"]].merge(
        p.drop(columns=["dong_name", "region_full_name", "gu"]), on="dong_code",
        how="left", validate="one_to_one")
    m["elderly_ratio"] = (m["elderly_population"] / m["total_population"] * 100).round(2)
    m["elderly_density"] = (m["elderly_population"] / m["area_km2"]).round(2)
    m["bus_stop_count"] = m["dong_code"].map(stops).fillna(0).astype(int)
    # 의료기관 수는 지오코딩 전이라 산출하지 않는다 (접근성은 accessibility.py에서 좌표 기반 계산)
    m["medical_facility_count"] = np.nan
    cols = ["dong_code", "dong_name", "adm_nm_full", "total_population", "elderly_population",
            "area_km2", "elderly_ratio", "elderly_density", "bus_stop_count", "medical_facility_count"]
    master = gpd.GeoDataFrame(m[cols + ["geometry"]], geometry="geometry", crs=boundary.crs)
    if C.DONG_MASTER.exists():
        C.DONG_MASTER.unlink()
    master.to_file(C.DONG_MASTER, driver="GeoJSON", encoding="utf-8")
    feat = pd.DataFrame(master.drop(columns=["geometry", "adm_nm_full"]))
    feat.to_csv(C.DONG_FEATURES, index=False, encoding="utf-8-sig")
    report.append(f"- 저장: data/processed/seongnam_dong_master.geojson, "
                  f"data/processed/dong_features_base.csv ({len(feat)}행)")
    report.append(f"- 버스정류장 동별 합계: {int(feat['bus_stop_count'].sum())} (matched 정류장 수와 일치해야 함)")
    return master


def run_preprocessing() -> dict:
    for d in [C.PROCESSED_DIR, C.OUTPUTS]:
        d.mkdir(parents=True, exist_ok=True)
    unmatched: list[dict] = []
    rep = ["# 전처리 리포트 (src/preprocessing.py 자동 생성)", "",
           "원본 위치: `data/raw/` (읽기 전용, 무수정)", ""]

    boundary = load_boundary()
    rep += ["## 0. 행정동 경계(GeoJSON) 로드",
            f"- 성남시 행정동 폴리곤 개수: {len(boundary)}", f"- CRS: {boundary.crs}", ""]
    rep.append("## 1. 고령인구현황 처리")
    pop = clean_population(unmatched, rep)
    rep += ["", "## 2. 버스정류장 처리"]
    bus, exc = clean_bus(boundary, unmatched, rep)
    rep += ["", "## 3. 의료기관 처리"]
    med = clean_medical(unmatched, rep)
    rep += ["", "## 4. 재가장기요양기관 (선택적 보조 데이터)"]
    care = clean_care(rep)
    rep += ["", "## 5. 동 단위 마스터"]
    master = build_dong_features(boundary, pop, bus, rep)

    pd.DataFrame(unmatched, columns=["source_dataset", "identifier", "issue_type", "reason"]) \
        .to_csv(C.OUTPUTS / "unmatched_records.csv", index=False, encoding="utf-8-sig")
    rep += ["", "## 6. 미매칭·예외 기록",
            f"- outputs/unmatched_records.csv: {len(unmatched)}건",
            f"- outputs/boundary_exceptions.csv: {len(exc)}건"]
    (C.OUTPUTS / "preprocessing_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    summary = {"n_dong": len(master), "n_bus": len(bus),
               "bus_matched": int((bus["join_status"] == "matched").sum()),
               "n_medical_rows": len(med), "n_medical_unique": int(med["unique_facility_id"].nunique()),
               "n_care_valid": len(care), "n_care_facility_based": int(care["facility_based"].sum()),
               "n_unmatched_records": len(unmatched)}
    print(f"preprocessing: {summary}")
    return summary


if __name__ == "__main__":
    run_preprocessing()
