"""전처리 재현성·회귀 테스트 — raw에서 다시 만든 가공 데이터가 저장소 버전과 동일해야 한다."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd

BASE = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ["population_clean.csv", "bus_stops_clean.csv", "medical_facilities_clean.csv",
                 "dong_features_base.csv", "care_facilities_clean.csv"]
OUTPUT_CSV = ["geocoding_targets_medical.csv", "geocoding_targets_care.csv", "unmatched_records.csv",
              "boundary_exceptions.csv"]


def test_preprocessing_reproduces_committed_data():
    tmp = BASE / "outputs_validation" / "_pytest_preprocessing"
    shutil.rmtree(tmp, ignore_errors=True)
    env = dict(os.environ, MOBILENS_OUT_ROOT=str(tmp), PYTHONIOENCODING="utf-8")
    subprocess.run([sys.executable, "-m", "src.preprocessing"], cwd=BASE, env=env, check=True, capture_output=True)
    for rel in [f"data/processed/{f}" for f in PROCESSED_CSV] + [f"outputs/{f}" for f in OUTPUT_CSV]:
        old = (BASE / rel).read_bytes().replace(b"\r\n", b"\n")
        new = (tmp / rel).read_bytes().replace(b"\r\n", b"\n")
        assert old == new, rel
    a = gpd.read_file(BASE / "data/processed/seongnam_dong_master.geojson")
    b = gpd.read_file(tmp / "data/processed/seongnam_dong_master.geojson")
    assert list(a.columns) == list(b.columns)
    da, db = a.drop(columns="geometry"), b.drop(columns="geometry")
    assert da.isna().equals(db.isna())
    assert (da.fillna("<NA>").astype(str) == db.fillna("<NA>").astype(str)).all().all()
    assert a.geometry.geom_equals_exact(b.geometry, tolerance=0).all()
