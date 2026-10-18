"""공간통계 — Global / Local Moran's I (LISA).

공간가중치: Queen contiguity (경계 또는 꼭짓점 공유 = 이웃), 행 표준화(row-standardized).
검증: 섬(이웃 0개) 없음, 연결요소 1개, 이웃관계 대칭, geometry 유효.
유의성: 조건부 순열검정 MORAN_PERMUTATIONS회 (seed=MORAN_SEED 고정 → 재실행 시 동일 p값),
        p_sim < MORAN_ALPHA 인 경우만 HH/HL/LH/LL로 분류, 나머지는 Not Significant.
사분면(q): 1=HH(자신·이웃평균 모두 평균 이상), 2=LH, 3=LL, 4=HL.

질문 "이 지역의 높은 MVI가 공간적으로 유의하게 모여 있는가?"에 대한 답:
  Global Moran I (전체 군집 여부) + Local Moran HH & p<α (어느 동이 고-고 군집인가).
출력: outputs/tables/local_moran_results.csv, outputs/json/spatial_summary.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen, W
from esda.moran import Moran, Moran_Local

from src import config as C

QUAD = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}


def build_weights(dong: gpd.GeoDataFrame) -> tuple[W, dict]:
    """Queen 가중치 생성 + 구조 검증. 문제가 있으면 예외."""
    if not dong.geometry.is_valid.all():
        raise ValueError("행정동 geometry 유효하지 않음")
    w = Queen.from_dataframe(dong, use_index=False)
    if w.islands:
        raise ValueError(f"이웃이 없는 섬 행정동 존재: {w.islands}")
    n_comp = int(w.n_components)
    if n_comp != 1:
        raise ValueError(f"공간가중치 연결요소 {n_comp}개 (1개 기대)")
    sym = all(i in w.neighbors[j] for i, nb in w.neighbors.items() for j in nb)
    if not sym:
        raise ValueError("Queen 이웃관계 비대칭")
    card = np.array(list(w.cardinalities.values()))
    info = {"type": "Queen contiguity, row-standardized", "n": int(w.n), "islands": [],
            "components": n_comp, "symmetric": sym, "neighbors_min": int(card.min()),
            "neighbors_mean": round(float(card.mean()), 2), "neighbors_max": int(card.max())}
    w.transform = "r"
    return w, info


def moran_local(y: np.ndarray, w: W, permutations: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Local Moran — 동별 I, p_sim, 사분면, 유의 분류."""
    lm = Moran_Local(np.asarray(y, float), w, permutations=permutations or C.MORAN_PERMUTATIONS,
                     seed=C.MORAN_SEED if seed is None else seed)
    sig = lm.p_sim < C.MORAN_ALPHA
    return pd.DataFrame({"moran_I": np.round(lm.Is, 4), "moran_p": np.round(lm.p_sim, 4),
                         "moran_quad": [QUAD[q] for q in lm.q],
                         "moran_class": [QUAD[q] if s else "Not Significant" for q, s in zip(lm.q, sig)],
                         "significant": sig})


def run_spatial_stats() -> pd.DataFrame:
    dong = gpd.read_file(C.DONG_MASTER, encoding="utf-8")
    dong["dong_code"] = dong["dong_code"].astype(str)
    w, winfo = build_weights(dong)
    print(f"spatial weights: {winfo}")
    mvi = pd.read_csv(C.PROCESSED_DIR / "mvi_results.csv", encoding="utf-8-sig", dtype={"dong_code": str})
    order = dong["dong_code"]
    frames, glob = [], {}
    for scen in C.MVI_SCENARIOS:
        y = mvi.set_index("dong_code").loc[order, f"mvi_{scen}"].to_numpy(float)
        f = moran_local(y, w)
        f.insert(0, "dong_code", order.values)
        f.insert(1, "dong_name", dong["dong_name"].values)
        f["scenario"] = scen
        frames.append(f)
        np.random.seed(C.MORAN_SEED)  # esda.Moran은 seed 인자가 없어 전역 난수 상태를 고정
        g = Moran(y, w, permutations=C.MORAN_PERMUTATIONS)
        glob[scen] = {"I": round(float(g.I), 4), "EI": round(float(g.EI), 4), "z_sim": round(float(g.z_sim), 3),
                      "p_sim": round(float(g.p_sim), 4),
                      "n_HH_significant": int((f["moran_class"] == "High-High").sum()),
                      "n_significant": int(f["significant"].sum())}
        print(f"{scen}: Global Moran I={g.I:.3f} p_sim={g.p_sim:.3f} | HH={glob[scen]['n_HH_significant']}")
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(C.TABLES / "local_moran_results.csv", index=False, encoding="utf-8-sig")
    summary = {"weights": winfo, "permutations": C.MORAN_PERMUTATIONS, "alpha": C.MORAN_ALPHA,
               "seed": C.MORAN_SEED, "global_moran": glob,
               "interpretation": "Global I>0 & p_sim<α: MVI가 공간적으로 군집. HH&유의: 해당 동과 이웃이 함께 높은 취약 군집."}
    (C.JSON_DIR / "spatial_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    return res


if __name__ == "__main__":
    run_spatial_stats()
