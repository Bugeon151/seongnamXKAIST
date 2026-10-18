"""데이터 검증 테스트 — 정상 원본은 PASS, 고의로 손상시킨 사본은 명확한 사유와 함께 FAIL."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from src import config as C
from src.data_validation import validate_raw, DataValidationError

WORK = Path(__file__).resolve().parent.parent / "outputs_validation" / "_pytest_raw"


def _copy(key: str) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    src = getattr(C, key)
    dst = WORK / src.name
    shutil.copyfile(src, dst)
    return dst


def test_raw_data_passes():
    rep = validate_raw()
    assert not rep.failures()
    warns = {r["check"] for r in rep.rows if r["level"] == "WARN"}
    assert "bus_missing_coords" in warns  # 알려진 원본 특성은 숨기지 않고 WARN으로 보고


def test_swapped_coordinates_detected():
    p = _copy("RAW_BUS")
    df = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
    df.loc[0, ["위도", "경도"]] = df.loc[0, ["경도", "위도"]].values
    df.to_csv(p, index=False, encoding="utf-8-sig")
    with pytest.raises(DataValidationError, match="bus_coords_in_korea"):
        validate_raw({"RAW_BUS": p})


def test_missing_column_detected():
    p = _copy("RAW_MEDICAL")
    pd.read_csv(p, encoding="utf-8-sig", dtype=str).drop(columns=["데이터기준일자"]).to_csv(
        p, index=False, encoding="utf-8-sig")
    with pytest.raises(DataValidationError, match="columns"):
        validate_raw({"RAW_MEDICAL": p})


def test_reference_date_mismatch_detected():
    p = _copy("RAW_CARE")
    df = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
    df.loc[df["데이터기준일"].notna(), "데이터기준일"] = "2025-01-01"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    with pytest.raises(DataValidationError, match="care_reference_date"):
        validate_raw({"RAW_CARE": p})


def test_population_integrity_detected():
    p = _copy("RAW_POPULATION")
    df = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
    col = f"{C.BASE_MONTH}_65세이상전체"
    df.loc[5, col] = "9,999,999"  # 65세 이상 > 전체
    df.to_csv(p, index=False, encoding="utf-8-sig")
    with pytest.raises(DataValidationError, match="population_values_valid"):
        validate_raw({"RAW_POPULATION": p})


def test_non_utf8_detected():
    p = _copy("RAW_AREA")
    text = p.read_text(encoding="utf-8-sig")
    p.write_bytes(text.encode("cp949"))
    with pytest.raises(DataValidationError, match="encoding_utf8"):
        validate_raw({"RAW_AREA": p})


def test_missing_input_file_detected():
    with pytest.raises(DataValidationError, match="exists:"):
        validate_raw({"RAW_BUS": WORK / "no_such_file.csv"})


def test_corrupted_raw_detected_by_sha256():
    # 값 하나만 바꿔도(스키마·범위 검사는 통과하는 수준) 커밋된 기준 해시와 달라 FAIL이어야 한다
    p = _copy("RAW_AREA")
    df = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
    col = "면적(제곱킬로미터)"
    df.loc[0, col] = str(float(df.loc[0, col]) + 0.01)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    with pytest.raises(DataValidationError, match="raw_sha256_manifest"):
        validate_raw({"RAW_AREA": p})


def test_manifest_covers_all_raw_files():
    from src.data_validation import MANIFEST, sha256_file
    listed = {ln.split("  ", 1)[1]: ln.split("  ", 1)[0]
              for ln in MANIFEST.read_text(encoding="utf-8").splitlines() if ln.strip()}
    raw = {p.relative_to(C.BASE_DIR).as_posix() for p in C.RAW_DIR.rglob("*") if p.is_file()}
    assert raw <= set(listed)
    # 줄바꿈 정규화 해시: CRLF/LF 표기만 다른 사본은 같은 해시
    WORK.mkdir(parents=True, exist_ok=True)
    a, b = WORK / "eol_a.txt", WORK / "eol_b.txt"
    a.write_bytes(b"x,y\r\n1,2\r\n")
    b.write_bytes(b"x,y\n1,2\n")
    assert sha256_file(a) == sha256_file(b)


def test_malformed_population_value_detected():
    p = _copy("RAW_POPULATION")
    df = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
    df.loc[3, f"{C.BASE_MONTH}_전체"] = "N/A"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    with pytest.raises(DataValidationError, match="population_values_valid"):
        validate_raw({"RAW_POPULATION": p})
