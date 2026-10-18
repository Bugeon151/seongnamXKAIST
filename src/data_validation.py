"""데이터 검증 — 원본 무수정 원칙, 수치 기준 PASS / WARN / FAIL.

validate_raw()       파이프라인 1단계. data/raw 원본의 존재·인코딩·컬럼·기준일·좌표·CRS·
                     결측·중복·코드 매칭·비정상 값 검사 (전처리 전에 실행)
                     + 원본 무결성: data/SHA256SUMS(커밋된 기준 해시)와 실제 파일 SHA-256 대조
validate_processed() 전처리 직후. 가공 데이터 무결성 + 지오코딩 결과(외부 API 산출물)와
                     의료기관 고유시설 ID의 1:1 연결 검사
원본을 의도적으로 갱신했다면 `python -m src.data_validation --write-manifest`로 기준 해시를 다시 만든다.
FAIL이 하나라도 있으면 DataValidationError로 모든 실패 항목을 한 번에 보고한다.
WARN은 알려진 원본 특성(좌표 결측 1건 등)으로, 숨기지 않고 출력·기록한다.
결과: outputs/tables/data_validation.csv
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

from src import config as C

BBOX_MARGIN_DEG = 0.005  # 성남시 경계 bbox 여유(약 500m) — 경계 반올림 오차 허용
MAX_MISSING_COORD_RATIO = 0.01  # 버스정류장 좌표 결측 허용 비율

REQUIRED_COLUMNS = {
    "RAW_POPULATION": ["행정구역", f"{C.BASE_MONTH}_전체", f"{C.BASE_MONTH}_65세이상전체"],
    "RAW_BUS": ["행정동", "정류장번호(ID)", "정류장명", "위도", "경도"],
    "RAW_MEDICAL": ["구별", "의료기관명", "의료기관종별", "의료기관주소(도로명)", "데이터기준일자"],
    "RAW_CARE": ["기관명", "소재지", "데이터기준일"],
    "RAW_AREA": ["구청명", "동 명", "면적(제곱킬로미터)"],
}


MANIFEST = C.BASE_DIR / "data" / "SHA256SUMS"
# 기준 해시 대상: 원본 전체 + 외부 API 결과(지오코딩)로 버전관리되는 입력
MANIFEST_EXTRA = [C.MEDICAL_GEOCODED, C.CARE_GEOCODED, C.DEPOT_GEOCODED]


class DataValidationError(RuntimeError):
    """검증 FAIL 항목이 있을 때 발생 — 메시지에 실패 항목 전체를 담는다."""


class ValidationReport:
    def __init__(self, stage: str):
        self.stage = stage
        self.rows: list[dict] = []

    def add(self, check: str, ok: bool, detail: str = "", warn: bool = False) -> bool:
        level = "PASS" if ok else ("WARN" if warn else "FAIL")
        self.rows.append({"stage": self.stage, "check": check, "level": level,
                          "pass": bool(ok), "detail": detail})
        print(f"  [{level}] {check} {detail}")
        return bool(ok)

    def failures(self) -> list[dict]:
        return [r for r in self.rows if r["level"] == "FAIL"]

    def raise_if_failed(self) -> None:
        f = self.failures()
        if f:
            msg = "; ".join(f"{r['check']} ({r['detail']})" for r in f)
            raise DataValidationError(f"[{self.stage}] 검증 실패 {len(f)}건: {msg}")


def _save(report: ValidationReport) -> None:
    C.TABLES.mkdir(parents=True, exist_ok=True)
    path = C.TABLES / "data_validation.csv"
    new = pd.DataFrame(report.rows)
    if path.exists():
        old = pd.read_csv(path, encoding="utf-8-sig")
        if "stage" in old.columns:
            old = old[old["stage"] != report.stage]
            new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False, encoding="utf-8-sig")


def sha256_file(p: Path) -> str:
    """줄바꿈 정규화(CRLF→LF) 후 SHA-256.

    git core.autocrlf 설정에 따라 같은 커밋도 Windows(CRLF)·Linux/macOS(LF)에서 텍스트 파일 바이트가 달라지므로,
    정규화한 내용으로 해시해야 어느 환경의 checkout·ZIP에서도 같은 값이 나온다(이진 파일은 사실상 원본 그대로).
    """
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _manifest_targets() -> list[Path]:
    return sorted(p for p in C.RAW_DIR.rglob("*") if p.is_file()) + [p for p in MANIFEST_EXTRA if p.exists()]


def write_manifest() -> Path:
    """기준 해시 파일 생성 — 한 줄에 '<줄바꿈 정규화 SHA-256>  <저장소 상대경로>'."""
    lines = [f"{sha256_file(p)}  {p.relative_to(C.BASE_DIR).as_posix()}" for p in _manifest_targets()]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{MANIFEST} ({len(lines)} files)")
    return MANIFEST


def _check_manifest(rep: "ValidationReport", overrides: dict[str, Path]) -> None:
    """원본·입력 파일이 커밋된 기준 해시와 같은지 (원본 무수정 원칙의 실제 검증)."""
    if not MANIFEST.exists():
        rep.add("raw_sha256_manifest", False, f"{MANIFEST.name} 없음 — 원본 무결성 기준 부재")
        return
    entries = [ln.split("  ", 1) for ln in MANIFEST.read_text(encoding="utf-8").splitlines() if ln.strip()]
    listed = {rel for _, rel in entries}
    bad = []
    for sha, rel in entries:
        p = overrides.get(Path(rel).name, C.BASE_DIR / rel)
        if not p.exists():
            bad.append(f"{rel}: 없음")
        elif sha256_file(p) != sha:
            bad.append(f"{rel}: SHA-256 불일치")
    rep.add("raw_sha256_manifest", not bad, f"{len(entries)}개 파일 대조" + (f", 불일치 {bad[:5]}" if bad else ""))
    extra = [p.relative_to(C.BASE_DIR).as_posix() for p in C.RAW_DIR.rglob("*")
             if p.is_file() and p.relative_to(C.BASE_DIR).as_posix() not in listed]
    if extra:
        rep.add("raw_unlisted_files", False, f"기준 해시에 없는 원본 파일 {extra[:5]}", warn=True)


def _file_date(p: Path) -> str | None:
    m = re.search(r"_(\d{8})\.csv$", p.name)
    return f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}" if m else None


def validate_raw(paths: dict[str, Path] | None = None) -> ValidationReport:
    """원본 데이터 검증. paths로 경로를 바꿔 테스트할 수 있다(기본: config)."""
    P = {k: getattr(C, k) for k in list(REQUIRED_COLUMNS) + ["RAW_BOUNDARY"]}
    P.update(paths or {})
    rep = ValidationReport("raw")
    print("== DATA VALIDATION (raw) ==")

    missing = [k for k, p in P.items() if not Path(p).exists()]
    for k, p in P.items():
        rep.add(f"exists:{Path(p).name}", Path(p).exists())
    if missing:
        rep.raise_if_failed()

    frames: dict[str, pd.DataFrame] = {}
    for k in REQUIRED_COLUMNS:
        raw_bytes = Path(P[k]).read_bytes()
        try:
            raw_bytes.decode("utf-8-sig")
            rep.add(f"encoding_utf8:{Path(P[k]).name}", True)
        except UnicodeDecodeError as e:
            rep.add(f"encoding_utf8:{Path(P[k]).name}", False, f"UTF-8 디코딩 실패: {e}")
            continue
        df = pd.read_csv(P[k], encoding="utf-8-sig", dtype=str)
        lack = [c for c in REQUIRED_COLUMNS[k] if c not in df.columns]
        rep.add(f"columns:{Path(P[k]).name}", not lack, f"누락={lack}" if lack else "")
        n_empty = int(df.isna().all(axis=1).sum())
        if n_empty:
            rep.add(f"empty_rows:{Path(P[k]).name}", False, f"완전 공백행 {n_empty}건 (전처리에서 제거·기록)",
                    warn=True)
        if not lack:
            frames[k] = df.dropna(how="all")
    rep.raise_if_failed()

    # 행정동 경계
    b = gpd.read_file(P["RAW_BOUNDARY"], encoding="utf-8")
    rep.add("boundary_count", len(b) == C.N_DONG, f"n={len(b)} (기대 {C.N_DONG})")
    rep.add("boundary_crs_wgs84", b.crs is not None and b.crs.to_epsg() == 4326, f"crs={b.crs}")
    rep.add("boundary_geometry_valid", bool(b.geometry.is_valid.all()) and bool(b.geometry.notna().all()))
    codes_b = b["adm_cd2"].astype(str)
    rep.add("boundary_code_unique", codes_b.is_unique)
    rep.add("boundary_code_seongnam", bool(codes_b.str.startswith("4113").all()),
            "성남시 행정코드(4113) 외 코드 존재" if not codes_b.str.startswith("4113").all() else "")
    x0, y0, x1, y1 = b.total_bounds
    bbox = (x0 - BBOX_MARGIN_DEG, y0 - BBOX_MARGIN_DEG, x1 + BBOX_MARGIN_DEG, y1 + BBOX_MARGIN_DEG)

    # 고령인구
    pop = frames["RAW_POPULATION"]
    parsed = pop["행정구역"].str.extract(r"\((\d{10})\)\s*$")[0]
    rep.add("population_code_parse", bool(parsed.notna().all()),
            f"파싱 실패 {int(parsed.isna().sum())}행")
    dong = parsed[parsed.str[-5:] != "00000"]
    rep.add("population_dong_rows", len(dong) == C.N_DONG, f"n={len(dong)}")
    rep.add("population_code_unique", dong.is_unique)
    months = sorted({c.split("_")[0] for c in pop.columns if c != "행정구역"})
    rep.add("population_base_month_latest", months[-1] == C.BASE_MONTH,
            f"데이터 최신월={months[-1]}, 설정={C.BASE_MONTH}")
    vals = pop.drop(columns=["행정구역"])
    rep.add("population_no_missing", not vals.isna().any().any(), f"결측 {int(vals.isna().sum().sum())}칸")
    try:
        tot = pop[f"{C.BASE_MONTH}_전체"].str.replace(",", "", regex=False).astype(int)
        eld = pop[f"{C.BASE_MONTH}_65세이상전체"].str.replace(",", "", regex=False).astype(int)
        rep.add("population_values_valid", bool(((tot > 0) & (eld >= 0) & (eld <= tot)).all()),
                "0 < 65세이상 ≤ 전체 조건 위반 행 존재" if not ((tot > 0) & (eld >= 0) & (eld <= tot)).all() else "")
    except ValueError as e:
        rep.add("population_values_valid", False, f"숫자 변환 실패: {e}")
    match = set(dong) & set(codes_b)
    rep.add("population_boundary_match", len(match) == C.N_DONG, f"matched={len(match)}/{C.N_DONG}")

    # 행정구역 면적
    area = frames["RAW_AREA"]
    a = pd.to_numeric(area["면적(제곱킬로미터)"], errors="coerce")
    rep.add("area_rows", len(area) == C.N_DONG, f"n={len(area)}")
    rep.add("area_positive", bool((a > 0).all()), f"결측/0 이하 {int((~(a > 0)).sum())}건")
    rep.add("area_name_unique", not area.duplicated(["구청명", "동 명"]).any())
    pop_keys = set(zip(pop["행정구역"].str.split().str[-2], pop["행정구역"].str.extract(
        r"\s(\S+)\(\d{10}\)\s*$")[0]))
    area_keys = set(zip(area["구청명"].str.strip(), area["동 명"].str.strip()))
    n_am = len(area_keys & pop_keys)
    rep.add("area_population_match", n_am == C.N_DONG, f"(구, 동명) 매칭 {n_am}/{C.N_DONG}")

    # 버스정류장
    bus = frames["RAW_BUS"]
    lat = pd.to_numeric(bus["위도"].str.strip(), errors="coerce")
    lon = pd.to_numeric(bus["경도"].str.strip(), errors="coerce")
    has = lat.notna() & lon.notna()
    ratio = 1 - has.mean()
    rep.add("bus_missing_coords_ratio", ratio <= MAX_MISSING_COORD_RATIO,
            f"{int((~has).sum())}/{len(bus)} ({ratio:.2%}, 허용 {MAX_MISSING_COORD_RATIO:.0%})")
    if (~has).any() and ratio <= MAX_MISSING_COORD_RATIO:
        rep.add("bus_missing_coords", False, f"{int((~has).sum())}건 — 삭제하지 않고 unmatched로 기록",
                warn=True)
    in_kr = lat[has].between(*C.KOREA_LAT) & lon[has].between(*C.KOREA_LON)
    rep.add("bus_coords_in_korea", bool(in_kr.all()), f"범위 밖 {int((~in_kr).sum())}건 (위경도 뒤바뀜 등)")
    in_bb = lon[has].between(bbox[0], bbox[2]) & lat[has].between(bbox[1], bbox[3])
    rep.add("bus_coords_in_seongnam_bbox", bool(in_bb.all()),
            f"성남시 경계 bbox(+{BBOX_MARGIN_DEG}°) 밖 {int((~in_bb).sum())}건")
    ids = pd.to_numeric(bus["정류장번호(ID)"], errors="coerce").dropna()
    rep.add("bus_id_unique", ids.is_unique, f"중복 {int(ids.duplicated().sum())}건")
    dup_nc = int(bus[has].duplicated(["정류장명", "위도", "경도"]).sum())
    if dup_nc:
        rep.add("bus_same_name_coord", False, f"동일 (정류장명·좌표) {dup_nc}건 — ID가 달라 별도 정류장으로 유지",
                warn=True)

    # 의료기관
    med = frames["RAW_MEDICAL"]
    d_med = _file_date(Path(P["RAW_MEDICAL"]))
    dates = sorted(med["데이터기준일자"].dropna().unique().tolist())
    rep.add("medical_reference_date", dates == [d_med], f"컬럼={dates}, 파일명={d_med}")
    miss = int(med[["의료기관명", "의료기관주소(도로명)"]].isna().any(axis=1).sum())
    rep.add("medical_name_address_present", miss == 0, f"결측 {miss}건")
    dup = int(med.duplicated(["의료기관명", "의료기관주소(도로명)"], keep=False).sum())
    if dup:
        rep.add("medical_duplicate_name_address", False,
                f"{dup}행 — 원본 보존, 고유시설 ID로 통합", warn=True)

    # 재가장기요양기관
    care = frames["RAW_CARE"]
    d_care = _file_date(Path(P["RAW_CARE"]))
    cd = sorted(care["데이터기준일"].dropna().unique().tolist())
    rep.add("care_reference_date", cd == [d_care], f"컬럼={cd}, 파일명={d_care}")
    fb = care["급여종류_주야간보호"].eq("○") | care["급여종류_단기보호"].eq("○")
    cap = pd.to_numeric(care["주야간보호_단기보호정원"], errors="coerce")
    rep.add("care_facility_capacity_present", bool(cap[fb].notna().all() and (cap[fb] > 0).all()),
            f"시설이용형 {int(fb.sum())}개소 중 정원 결측/0 {int((~(cap[fb] > 0)).sum())}건")

    # OSM 도로망 스냅샷 (접근성·노선 거리)
    osm_meta = C.OSM_DIR / "osm_snapshot_meta.json"
    need = [C.OSM_DIR / f"seongnam_{k}_{t}.csv.gz" for k in ("walk", "drive") for t in ("nodes", "edges")]
    rep.add("osm_snapshot_files", osm_meta.exists() and all(p.exists() for p in need),
            "없으면 python scripts/download_osm_network.py (인터넷 필요)")
    if osm_meta.exists():
        import json
        om = json.loads(osm_meta.read_text(encoding="utf-8"))
        rep.add("osm_snapshot_license", "ODbL" in om.get("license", ""), f"downloaded_at={om.get('downloaded_at')}")

    _check_manifest(rep, {Path(getattr(C, k)).name: Path(v) for k, v in (paths or {}).items()})
    _save(rep)
    rep.raise_if_failed()
    print(f"DATA VALIDATION (raw): PASS (WARN {sum(r['level'] == 'WARN' for r in rep.rows)}건)")
    return rep


def validate_processed() -> ValidationReport:
    rep = ValidationReport("processed")
    print("== DATA VALIDATION (processed) ==")
    pop = pd.read_csv(C.DONG_FEATURES, encoding="utf-8-sig")
    rep.add("dong_count==50", len(pop) == C.N_DONG, f"n={len(pop)}")
    rep.add("dong_code_unique", pop["dong_code"].astype(str).is_unique)
    rep.add("features_no_missing", not pop.drop(columns=["medical_facility_count"]).isna().any().any())
    rep.add("elderly_ratio_range", bool(pop["elderly_ratio"].between(0, 100).all()))
    rep.add("elderly_density_positive", bool((pop["elderly_density"] > 0).all()))
    b = gpd.read_file(C.DONG_MASTER, encoding="utf-8")
    rep.add("master_50", len(b) == C.N_DONG, f"n={len(b)}")
    rep.add("master_geometry_valid", bool(b.geometry.is_valid.all()))
    rep.add("master_crs_wgs84", b.crs is not None and b.crs.to_epsg() == 4326, f"crs={b.crs}")
    m = set(pop["dong_code"].astype(str)) & set(b["dong_code"].astype(str))
    rep.add("features_master_match_100pct", len(m) == C.N_DONG, f"matched={len(m)}/{C.N_DONG}")
    bus = pd.read_csv(C.BUS_CLEAN, encoding="utf-8-sig")
    matched = int((bus["join_status"] == "matched").sum())
    rep.add("bus_spatial_join>=99pct", matched / len(bus) >= 0.99, f"{matched}/{len(bus)}")
    rep.add("bus_stop_count_consistent", int(pop["bus_stop_count"].sum()) == matched,
            f"동별 합={int(pop['bus_stop_count'].sum())}, matched={matched}")
    lat_ok = bus["위도"].dropna().between(*C.KOREA_LAT).all()
    lon_ok = bus["경도"].dropna().between(*C.KOREA_LON).all()
    rep.add("bus_coords_in_korea", bool(lat_ok and lon_ok))

    # 지오코딩 결과(외부 API 산출물)가 재생성된 의료기관 고유시설 ID와 1:1로 연결되는지
    if not C.MEDICAL_GEOCODED.exists():
        rep.add("geocoded_file_exists", False,
                f"{C.MEDICAL_GEOCODED.name} 없음 — python -m src.geocoding 실행 필요(Nominatim, 30~90분)")
    else:
        geo = pd.read_csv(C.MEDICAL_GEOCODED, encoding="utf-8-sig", dtype=str)
        med = pd.read_csv(C.MEDICAL_CLEAN, encoding="utf-8-sig", dtype=str) \
            .drop_duplicates("unique_facility_id")
        rep.add("geocoded_ids==medical_unique_ids",
                set(geo["unique_facility_id"]) == set(med["unique_facility_id"]),
                f"geocoded={geo['unique_facility_id'].nunique()}, medical={len(med)}")
        j = geo.merge(med, on="unique_facility_id", how="inner")
        same = (j["facility_name"] == j["의료기관명"]) & (j["raw_address"] == j["의료기관주소(도로명)"])
        rep.add("geocoded_name_address_match", bool(same.all()), f"mismatch={int((~same).sum())}")
        ok = geo[geo["geocode_status"] == "success"]
        la, lo = ok["latitude"].astype(float), ok["longitude"].astype(float)
        rep.add("geocoded_coords_in_korea", bool(la.between(*C.KOREA_LAT).all() and lo.between(*C.KOREA_LON).all()))
        n_in = int((ok["in_seongnam"].astype(str) == "True").sum())
        rep.add("geocoded_in_seongnam_count", n_in > 0, f"성남시 폴리곤 내 {n_in}/{len(geo)}")
    if not C.CARE_GEOCODED.exists():
        rep.add("care_geocoded_file_exists", False,
                f"{C.CARE_GEOCODED.name} 없음 — python -m src.geocoding care 실행 필요(Nominatim)")
    else:
        cg = pd.read_csv(C.CARE_GEOCODED, encoding="utf-8-sig", dtype=str)
        cc = pd.read_csv(C.CARE_CLEAN, encoding="utf-8-sig", dtype=str)
        rep.add("care_geocoded_ids==care_ids", set(cg["care_id"]) == set(cc["care_id"]),
                f"geocoded={cg['care_id'].nunique()}, care={len(cc)}")
        j = cg.merge(cc, on="care_id", suffixes=("_g", ""))
        rep.add("care_geocoded_address_match", bool((j["raw_address_g"] == j["raw_address"]).all()))
        fb = j[j["facility_based"] == "True"]
        n_in = int((fb["in_seongnam"] == "True").sum())
        rep.add("care_facility_based_geocoded", n_in > 0, f"시설이용형 {n_in}/{len(fb)} 성남시 내 좌표")
        if n_in < len(fb):
            rep.add("care_facility_based_missing", False,
                    f"{len(fb) - n_in}개소 좌표 없음/시 경계 밖 — 복지 공급 과소추정 가능", warn=True)
    _save(rep)
    rep.raise_if_failed()
    print("DATA VALIDATION (processed): PASS")
    return rep


def run_validation() -> bool:
    """하위호환: 가공 데이터 검증 (실패 시 DataValidationError)."""
    validate_processed()
    return True


if __name__ == "__main__":
    if "--write-manifest" in sys.argv:
        write_manifest()
    else:
        validate_raw()
        validate_processed()
