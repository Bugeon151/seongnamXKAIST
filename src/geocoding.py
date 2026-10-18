"""의료기관·재가장기요양기관 지오코딩 — 실제 geocoder만 사용, 가짜 좌표 생성 금지.

실행: python -m src.geocoding          (의료기관 1차 패스)
      python -m src.geocoding bounded  (의료기관 2차 bbox 패스)
      python -m src.geocoding care     (재가장기요양기관, 1·2차 통합)
결과 파일(data/processed/*_geocoded.csv)은 외부 API 실측값이라 버전관리되는 입력으로 취급한다.

주소 정규화 4단계 attempt를 기록하고, 성공 좌표는 수치·국내 범위·
성남시 폴리곤 검사를 거쳐 행정동 spatial join까지 수행한다.
API KEY가 없어도 Nominatim(무료·키 불필요)으로 진행하고,
외부 유료 키가 필요한 서비스는 키가 있을 때만 시도한다.
중단 후 재실행하면 캐시부터 이어서 처리한다(resume).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from src import config as C

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim 이용정책: 식별 가능한 User-Agent 필요 — 연락처는 환경변수 NOMINATIM_CONTACT로 지정
UA = {"User-Agent": "MobiLens-Seongnam-research/1.0 (academic project"
                    + (f"; {os.environ['NOMINATIM_CONTACT']}" if os.environ.get("NOMINATIM_CONTACT") else "")
                    + ")"}
CACHE_PATH = C.BASE_DIR / "outputs" / "logs" / "geocode_cache" / "nominatim_cache.json"


def normalize_attempts(raw: str) -> list[tuple[str, str]]:
    """원본 주소 → 최대 4단계 정규화 시도 문구 목록."""
    out: list[tuple[str, str]] = []
    a0 = (raw or "").strip()
    if a0:
        out.append(("L0_raw", a0))
    # L1: 괄호 내부 부가정보 제거
    a1 = re.sub(r"\([^)]*\)", " ", a0).strip()
    a1 = re.sub(r"\s+", " ", a1)
    if a1 and a1 != a0:
        out.append(("L1_no_paren", a1))
    # L2: 층/호 상세주소 제거
    a2 = re.sub(r"\s\d+층.*$", "", a1).strip()
    a2 = re.sub(r",\s*[^,]*?(호|층)\s*[^,]*$", "", a2).strip()
    a2 = re.sub(r"\s+", " ", a2)
    if a2 and a2 not in (a0, a1):
        out.append(("L2_no_floor", a2))
    # L3: 도로명 핵심부 (도로명+건물번호까지) 추출
    m = re.search(r"(.+?[로길])\s*(\d+(?:-\d+)?)", a2)
    if m:
        a3 = m.group(0).strip()
        # 시/구 접두 유지
        prefix = re.match(r"(경기도?\s+성남시\s+\S+구)", a2)
        if prefix and not a3.startswith("경기"):
            a3 = prefix.group(1) + " " + re.sub(r"^.*?([가-힣]+[로길]\s*\d.*)$", r"\1", a3)
        if a3 and a3 not in (a0, a1, a2):
            out.append(("L3_road_core", a3))
    return out


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"WARNING: 지오코딩 캐시 손상({e}) — 빈 캐시로 다시 조회합니다: {CACHE_PATH}")
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def nominatim_geocode(query: str, cache: dict, bounded: bool = False) -> dict | None:
    """Nominatim 실측 지오코딩. 캐시 우선, 1.1초 throttle 준수.

    bounded=True면 성남시 bbox로 viewbox를 강제(bounded=1)하여
    타 도시 오매칭을 차단한다. 캐시 키를 분리한다.
    """
    key = ("BBOX||" if bounded else "") + query
    cached = cache.get(key)
    # 결과 없음(NO_RESULT)은 확정값으로 재사용, 네트워크 오류는 일시적이므로 재조회
    if cached is not None and cached.get("_error", "NO_RESULT") == "NO_RESULT":
        return None if "_error" in cached else cached
    params = {"format": "json", "q": query, "limit": 1, "countrycodes": "kr",
              "accept-language": "ko", "addressdetails": 1}
    if bounded:
        # 성남시 bbox: lon 127.03~127.19, lat 37.33~37.48 (버스정류장 실측 범위 기반)
        params.update({"viewbox": "127.03,37.48,127.19,37.33", "bounded": 1})
    req = urllib.request.Request(
        NOMINATIM_URL + "?" + urllib.parse.urlencode(params), headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        cache[key] = {"_error": f"{type(e).__name__}: {e}"}
        _save_cache(cache)
        time.sleep(1.1)
        return None
    time.sleep(1.1)  # Nominatim 이용정책: 초당 1건 이하
    if not data:
        cache[key] = {"_error": "NO_RESULT"}
        _save_cache(cache)
        return None
    d = data[0]
    rec = {"lat": d.get("lat"), "lon": d.get("lon"), "display_name": d.get("display_name", "")}
    cache[key] = rec
    _save_cache(cache)
    return rec


def run_geocoding(limit: int | None = None, throttle: float = 1.1) -> dict:
    """전건 지오코딩 실행. 결과 dict(성공률 등) 반환 + 파일 저장."""
    targets = pd.read_csv(C.GEOCODE_TARGETS, encoding="utf-8-sig")
    if limit:
        targets = targets.head(limit)
    boundary = gpd.read_file(C.RAW_BOUNDARY, encoding="utf-8")
    boundary_metric = boundary.to_crs(C.CRS_METRIC)

    cache = _load_cache()
    rows: list = []
    n_success_by_level: dict[str, int] = {}
    n_attempt = 0
    n_done = 0
    n_total = len(targets)
    for _, t in targets.iterrows():
        n_done += 1
        if n_done % 200 == 0 or n_done == n_total:
            print(f"progress {n_done}/{n_total}", flush=True)
        fid = t["unique_facility_id"]
        raw_addr = str(t.get("의료기관주소(도로명)", ""))
        name = str(t.get("의료기관명", ""))
        rec = {"unique_facility_id": fid, "facility_name": name,
               "raw_address": raw_addr, "latitude": None, "longitude": None,
               "geocoder": "nominatim", "geocode_status": "failed",
               "geocode_attempt": None, "returned_address": None}
        for level, q in normalize_attempts(raw_addr):
            n_attempt += 1
            hit = nominatim_geocode(q, cache)
            if hit and "_error" not in hit:
                try:
                    lat, lon = float(hit["lat"]), float(hit["lon"])
                except (TypeError, ValueError):
                    continue
                rec.update({"latitude": lat, "longitude": lon,
                            "geocode_status": "success", "geocode_attempt": level,
                            "returned_address": hit.get("display_name")})
                n_success_by_level[level] = n_success_by_level.get(level, 0) + 1
                break
        else:
            rec["geocode_attempt"] = "all_failed"
        rows.append(rec)

    geo = pd.DataFrame(rows)
    return _finalize(geo, n_success_by_level)


def _finalize(geo: pd.DataFrame, n_success_by_level: dict,
              extra: dict | None = None) -> dict:
    """검증→폴리곤 검사→파일 저장→요약. 1차/2차 패스 공용."""
    boundary = gpd.read_file(C.RAW_BOUNDARY, encoding="utf-8")
    boundary_metric = boundary.to_crs(C.CRS_METRIC)

    # --- 검증: 수치·국내 범위 ---
    def _valid(r):
        try:
            lat, lon = float(r["latitude"]), float(r["longitude"])
        except (TypeError, ValueError):
            return False
        return (C.KOREA_LAT[0] <= lat <= C.KOREA_LAT[1]
                and C.KOREA_LON[0] <= lon <= C.KOREA_LON[1])
    geo["in_korea_bbox"] = geo.apply(
        lambda r: _valid(r) if r["geocode_status"] == "success" else False, axis=1)
    geo.loc[geo["geocode_status"] == "success",
            "geocode_status"] = geo.loc[geo["geocode_status"] == "success"].apply(
        lambda r: "success" if r["in_korea_bbox"] else "invalid_coords", axis=1)

    # --- 성남시 폴리곤 검사 + 행정동 spatial join ---
    ok = geo[geo["geocode_status"] == "success"].copy()
    ok["geometry"] = ok.apply(lambda r: Point(float(r["longitude"]), float(r["latitude"])), axis=1)
    ok_gdf = gpd.GeoDataFrame(ok, geometry="geometry", crs=C.CRS_WGS84)
    joined = gpd.sjoin(ok_gdf, boundary[["adm_cd2", "adm_nm", "geometry"]],
                       how="left", predicate="within")
    geo["in_seongnam"] = False
    geo["official_dong_code"] = None
    geo["official_dong_name"] = None
    geo.loc[geo["geocode_status"] == "success", "in_seongnam"] = (
        joined["adm_cd2"].notna().values)
    geo.loc[geo["geocode_status"] == "success", "official_dong_code"] = (
        joined["adm_cd2"].values)
    geo.loc[geo["geocode_status"] == "success", "official_dong_name"] = (
        joined["adm_nm"].values)

    # 폴리곤 밖 좌표 → 최근접 경계 거리 계산 (삭제하지 않음)
    outside = geo[(geo["geocode_status"] == "success") & (~geo["in_seongnam"])].copy()
    exceptions = []
    if len(outside):
        out_gdf = gpd.GeoDataFrame(
            outside,
            geometry=gpd.points_from_xy(outside["longitude"].astype(float),
                                        outside["latitude"].astype(float)),
            crs=C.CRS_WGS84).to_crs(C.CRS_METRIC)
        for idx, row in out_gdf.iterrows():
            dist = float(boundary_metric.distance(row.geometry).min())
            exceptions.append({"unique_facility_id": row["unique_facility_id"],
                               "facility_name": row["facility_name"],
                               "latitude": row["latitude"], "longitude": row["longitude"],
                               "nearest_boundary_m": round(dist, 1)})
    exc_df = pd.DataFrame(exceptions)
    exc_df.to_csv(C.OUTPUTS / "medical_boundary_exceptions.csv",
                  index=False, encoding="utf-8-sig")

    failures = geo[geo["geocode_status"] != "success"].copy()
    failures.to_csv(C.OUTPUTS / "geocoding_failures_medical.csv",
                    index=False, encoding="utf-8-sig")
    geo.to_csv(C.MEDICAL_GEOCODED, index=False, encoding="utf-8-sig")

    n_ok = int((geo["geocode_status"] == "success").sum())
    n_in = int(geo["in_seongnam"].sum())
    summary = {
        "n_targets": len(geo), "n_success": n_ok,
        "success_rate": round(n_ok / len(geo), 4) if len(geo) else 0.0,
        "n_in_seongnam": n_in,
        "dong_join_rate_on_success": round(n_in / n_ok, 4) if n_ok else 0.0,
        "n_outside": len(exc_df), "n_failed": len(failures),
        "success_by_level": n_success_by_level,
    }
    if extra:
        summary.update(extra)
    (C.TABLES / "geocoding_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_bounded_pass() -> dict:
    """2차 패스: 1차에서 실패/폴리곤 밖이었던 목표를 성남 bbox bounded로 재지오코딩.

    1차 결과(C.MEDICAL_GEOCODED)를 logs에 백업한 뒤 해당 행만 갱신한다.
    """
    import shutil
    shutil.copy(C.MEDICAL_GEOCODED, C.LOGS / "medical_geocoded_pass1.csv")
    geo = pd.read_csv(C.MEDICAL_GEOCODED, encoding="utf-8-sig")
    need = geo[~((geo["geocode_status"] == "success") & (geo["in_seongnam"] == True))]  # noqa: E712
    print(f"bounded pass targets: {len(need)}/{len(geo)}", flush=True)
    cache = _load_cache()
    fixed = 0
    for n, (idx, r) in enumerate(need.iterrows(), 1):
        if n % 100 == 0 or n == len(need):
            print(f"bounded progress {n}/{len(need)} fixed={fixed}", flush=True)
        for level, q in normalize_attempts(str(r["raw_address"])):
            hit = nominatim_geocode(q, cache, bounded=True)
            if hit and "_error" not in hit:
                try:
                    lat, lon = float(hit["lat"]), float(hit["lon"])
                except (TypeError, ValueError):
                    continue
                geo.loc[idx, ["latitude", "longitude"]] = [lat, lon]
                geo.loc[idx, "geocoder"] = "nominatim-bounded"
                geo.loc[idx, "geocode_status"] = "success"
                geo.loc[idx, "geocode_attempt"] = "B_" + level
                geo.loc[idx, "returned_address"] = hit.get("display_name")
                fixed += 1
                break
        else:
            geo.loc[idx, ["latitude", "longitude"]] = [None, None]
            geo.loc[idx, "geocode_status"] = "failed"
            geo.loc[idx, "geocode_attempt"] = "bounded_all_failed"
    by_level: dict[str, int] = {}
    for v in geo["geocode_attempt"].dropna():
        by_level[str(v)] = by_level.get(str(v), 0) + 1
    summary = _finalize(geo, by_level, extra={"bounded_pass": True,
                        "bounded_retried": len(need), "bounded_fixed": fixed})
    return summary


def geocode_care() -> dict:
    """재가장기요양기관 지오코딩 (의료기관과 동일한 정규화·검증 규칙).

    기관별로 주소 정규화 단계(L0~L3)를 무제한 검색 → 성남 bbox 제한 검색 순으로 시도하고,
    성남시 폴리곤 안에 떨어진 첫 결과를 채택한다. 폴리곤 밖 결과만 있으면 좌표는 기록하되
    in_seongnam=False로 남긴다(접근성 계산에서 제외). 결과를 찾지 못하면 좌표를 만들지 않는다.
    """
    targets = pd.read_csv(C.GEOCODE_TARGETS_CARE, encoding="utf-8-sig")
    boundary = gpd.read_file(C.RAW_BOUNDARY, encoding="utf-8")
    cache = _load_cache()
    rows = []
    for n, (_, t) in enumerate(targets.iterrows(), 1):
        if n % 50 == 0 or n == len(targets):
            print(f"care progress {n}/{len(targets)}", flush=True)
        rec = {"care_id": t["care_id"], "facility_name": t["facility_name"],
               "raw_address": t["raw_address"], "latitude": None, "longitude": None,
               "geocoder": None, "geocode_status": "failed", "geocode_attempt": "all_failed",
               "returned_address": None, "in_seongnam": False,
               "official_dong_code": None, "official_dong_name": None}
        fallback = None
        for bounded in (False, True):
            for level, q in normalize_attempts(str(t["raw_address"])):
                hit = nominatim_geocode(q, cache, bounded=bounded)
                if not hit or "_error" in hit:
                    continue
                lat, lon = float(hit["lat"]), float(hit["lon"])
                inside = boundary[boundary.contains(Point(lon, lat))]
                cand = {"latitude": lat, "longitude": lon,
                        "geocoder": "nominatim-bounded" if bounded else "nominatim",
                        "geocode_status": "success",
                        "geocode_attempt": ("B_" if bounded else "") + level,
                        "returned_address": hit.get("display_name")}
                if len(inside):
                    cand.update({"in_seongnam": True,
                                 "official_dong_code": str(inside.iloc[0]["adm_cd2"]),
                                 "official_dong_name": inside.iloc[0]["adm_nm"].split()[-1]})
                    rec.update(cand)
                    break
                fallback = fallback or cand
            if rec["in_seongnam"]:
                break
        if not rec["in_seongnam"] and fallback:
            rec.update(fallback)
        rows.append(rec)
    geo = pd.DataFrame(rows)
    ok = geo["geocode_status"] == "success"
    in_sn = geo["in_seongnam"].astype(bool)
    geo.to_csv(C.CARE_GEOCODED, index=False, encoding="utf-8-sig")
    geo[~in_sn].to_csv(C.OUTPUTS / "geocoding_failures_care.csv", index=False, encoding="utf-8-sig")
    summary = {"n_targets": len(geo), "n_success": int(ok.sum()), "n_in_seongnam": int(in_sn.sum()),
               "n_outside_polygon": int((ok & ~in_sn).sum()), "n_failed": int((~ok).sum()),
               "success_by_level": geo.loc[in_sn, "geocode_attempt"].value_counts().to_dict()}
    (C.TABLES / "geocoding_report_care.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def geocode_depots() -> pd.DataFrame:
    """차량 거점(구 보건소) 중 의료기관 지오코딩에 실패한 곳을 기관명·주소로 재조회.

    기관명 POI 검색 → 주소 정규화 단계 순으로 성남 bbox 제한 검색하고, 성남시 폴리곤 안의 첫 결과만 채택.
    결과: data/processed/depots_geocoded.csv (쿼리·반환주소 기록). 찾지 못하면 좌표를 만들지 않고 예외.
    """
    med = pd.read_csv(C.MEDICAL_CLEAN, encoding="utf-8-sig").drop_duplicates("unique_facility_id")
    geo = pd.read_csv(C.MEDICAL_GEOCODED, encoding="utf-8-sig").set_index("unique_facility_id")
    boundary = gpd.read_file(C.RAW_BOUNDARY, encoding="utf-8")
    cache = _load_cache()
    rows = []
    for _, r in med[med["의료기관종별"] == "보건소"].iterrows():
        g = geo.loc[r["unique_facility_id"]]
        if g["geocode_status"] == "success" and str(g["in_seongnam"]) == "True":
            continue
        name = str(r["의료기관명"])
        addr = str(r["의료기관주소(도로명)"])
        paren = re.search(r"\(([^,)]+)", addr)
        spaced = re.sub(r"(\S)(보건소)", r"\1 \2", name.replace("성남시", ""))
        queries = [("NAME", name), ("NAME_CITY", f"성남시 {name}")]
        if paren:  # OSM POI명은 '수정구 보건소'처럼 띄어 쓰는 경우가 많아 '기관명 + 법정동'으로도 조회
            queries.append(("NAME_DONG", f"{spaced} {paren.group(1).strip()}"))
        queries += normalize_attempts(addr)
        rec = None
        for level, q in queries:
            hit = nominatim_geocode(q, cache, bounded=True)
            if not hit or "_error" in hit:
                continue
            if level.startswith("NAME") and "보건소" not in str(hit.get("display_name", "")):
                continue  # 기관명 검색은 반환 결과가 실제 보건소 POI일 때만 채택 (도로 등 오매칭 방지)
            lat, lon = float(hit["lat"]), float(hit["lon"])
            inside = boundary[boundary.contains(Point(lon, lat))]
            if len(inside):
                rec = {"unique_facility_id": r["unique_facility_id"], "gu": r["구별"], "depot_name": name,
                       "latitude": lat, "longitude": lon, "geocoder": "nominatim-bounded", "query_level": level,
                       "query": q, "returned_address": hit.get("display_name"), "in_seongnam": True,
                       "official_dong_name": inside.iloc[0]["adm_nm"].split()[-1]}
                break
        if rec is None:
            raise RuntimeError(f"차량 거점 {name} 지오코딩 실패 — 좌표를 임의로 만들지 않음")
        rows.append(rec)
        print(f"depot {name}: {rec['query_level']} → {rec['returned_address']}")
    out = pd.DataFrame(rows)
    out.to_csv(C.DEPOT_GEOCODED, index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "bounded":
        run_bounded_pass()
    elif len(sys.argv) > 1 and sys.argv[1] == "care":
        geocode_care()
    elif len(sys.argv) > 1 and sys.argv[1] == "depots":
        geocode_depots()
    else:
        lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
        run_geocoding(limit=lim)
