"""Robust Hotspot + 운영 묶음(DBSCAN).

Robust Hotspot = 강건성 격자(src/sensitivity.py, 36 시나리오)에서 High-High·유의(p<α)로
                 나타난 비율이 ROBUST_STABLE_SHARE 이상인 동 (stable).
불안정 hotspot = ROBUST_UNSTABLE_SHARE 이상 ~ stable 미만 (조건부 — 파라미터에 따라 달라짐).
DBSCAN은 Local Moran과 역할을 섞지 않는다: stable hotspot 동 중심점을 거리 eps 이내로
묶어 '함께 운영할 수 있는 권역'을 제시하는 보조 도구(유의성 판단에는 쓰지 않음).

출력: outputs/tables/robust_hotspots.csv, outputs/tables/robust_hotspot_meta.json,
      data/processed/robust_hotspots.geojson
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN

from src import config as C

DBSCAN_EPS_M = 2000
DBSCAN_MIN_SAMPLES = 2


def run_hotspot() -> dict:
    freq = pd.read_csv(C.TABLES / "hotspot_frequency.csv", encoding="utf-8-sig", dtype={"dong_code": str})
    stable = freq[freq["hotspot_class"] == "stable"]
    unstable = freq[freq["hotspot_class"] == "unstable"]
    robust_codes = sorted(stable["dong_code"])

    dong = gpd.read_file(C.DONG_MASTER, encoding="utf-8")
    dong["dong_code"] = dong["dong_code"].astype(str)
    dong = dong.merge(freq[["dong_code", "hh_share", "hotspot_class"]], on="dong_code", how="left",
                      validate="one_to_one")
    dong["robust_hotspot"] = dong["dong_code"].isin(robust_codes)

    hot = dong[dong["robust_hotspot"]].to_crs(C.CRS_METRIC)
    op_group: dict[str, str] = {}
    if len(hot) >= DBSCAN_MIN_SAMPLES:
        xy = np.column_stack([hot.geometry.centroid.x, hot.geometry.centroid.y])
        lab = DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(xy)
        op_group = {c: f"OP-{L + 1}" for c, L in zip(hot["dong_code"], lab) if L != -1}
    dong["operation_group"] = dong["dong_code"].map(op_group)

    stable.merge(dong[["dong_code", "operation_group"]], on="dong_code", how="left").to_csv(
        C.TABLES / "robust_hotspots.csv", index=False, encoding="utf-8-sig")
    dong.to_file(C.PROCESSED_DIR / "robust_hotspots.geojson", driver="GeoJSON", encoding="utf-8")

    name = dict(zip(dong["dong_code"], dong["dong_name"]))
    groups: dict[str, list[str]] = {}
    for c, g in op_group.items():
        groups.setdefault(g, []).append(name[c])
    meta = {"definition": (f"강건성 격자 시나리오 중 HH & p<{C.MORAN_ALPHA} 비율 ≥ {C.ROBUST_STABLE_SHARE}"
                           f" (불안정: ≥ {C.ROBUST_UNSTABLE_SHARE})"),
            "n_robust": len(robust_codes), "dong_codes": robust_codes,
            "dong_names": [name[c] for c in robust_codes],
            "unstable_codes": sorted(unstable["dong_code"]),
            "unstable_names": [name[c] for c in sorted(unstable["dong_code"])],
            "dbscan": {"eps_m": DBSCAN_EPS_M, "min_samples": DBSCAN_MIN_SAMPLES,
                       "role": "stable hotspot 운영 권역 묶음 (유의성 판단 아님)"},
            "operation_groups": {k: sorted(v) for k, v in sorted(groups.items())}}
    (C.TABLES / "robust_hotspot_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    print(f"robust hotspots (stable {len(robust_codes)}): {meta['dong_names']}; "
          f"unstable {len(meta['unstable_codes'])}")
    return meta


if __name__ == "__main__":
    run_hotspot()
