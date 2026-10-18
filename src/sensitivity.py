"""강건성(민감도) 분석 — 파라미터 격자 전체에서 취약 순위·hotspot이 유지되는지 검증.

격자 = MVI 가중치(MVI_SCENARIOS 3) × 거리감쇠 배율(DECAY_SCALE_VARIANTS 3)
       × 결합형식(곱/가중합 2) × 정규화(min-max/순위 2) = 36 시나리오.
각 시나리오마다 접근성(E2SFCA, 저장된 거리행렬 재사용) → MVI → Local Moran(고정 seed)을 다시 계산한다.

지표:
  appearance frequency  동별로 HH·유의로 나타난 시나리오 비율 / 상위 k 취약지역에 든 비율
  Jaccard               시나리오별 HH 집합의 primary 대비·쌍별 유사도
  rank correlation      시나리오별 MVI 순위의 primary 대비 Spearman ρ
분류: 비율 ≥ ROBUST_STABLE_SHARE → stable, ≥ ROBUST_UNSTABLE_SHARE → unstable, 그 외 none.
'No robustness'(primary 1개 시나리오만 본 hotspot)와 stable 집합의 차이도 기록한다.

출력: outputs/tables/robustness_grid.csv, outputs/tables/hotspot_frequency.csv,
      outputs/tables/robustness_report.json
"""
from __future__ import annotations

import json
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import config as C
from src.accessibility import load_inputs, compute_access_table, scaled_bands
from src.mvi import compute_components, combine, NEED_COLS
from src.spatial_statistics import build_weights, moran_local


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def scenario_grid() -> list[dict]:
    grid = []
    for (scen, (wn, wa)), scale, form, norm in product(C.MVI_SCENARIOS.items(), C.DECAY_SCALE_VARIANTS,
                                                        C.ROBUST_FORMS, C.ROBUST_NORMALIZATIONS):
        grid.append({"id": f"{scen}|decay{scale}|{form}|{norm}", "mvi_weights": scen, "w_need": wn,
                     "w_access": wa, "decay_scale": scale, "form": form, "normalization": norm})
    return grid


def run_sensitivity() -> dict:
    inp = load_inputs()
    npz = np.load(C.INTERIM_DIR / "distance_matrices.npz")
    mats = {k: npz[k] for k in npz.files}
    dong = inp.dong
    w, _ = build_weights(dong)
    feat = inp.features

    access_by_scale = {s: compute_access_table(inp, mats, scaled_bands(s)) for s in C.DECAY_SCALE_VARIANTS}
    grid = scenario_grid()
    ranks, hh_sets, top_sets, rows = {}, {}, {}, []
    for g in grid:
        acc = access_by_scale[g["decay_scale"]]
        df = pd.concat([feat[NEED_COLS].reset_index(drop=True),
                        acc[list(C.ACCESS_COMPONENTS.values())].reset_index(drop=True)], axis=1)
        comp = compute_components(df, normalization=g["normalization"])
        mvi = combine(comp["need"], comp["access_deficit"], g["w_need"], g["w_access"], g["form"])
        lm = moran_local(mvi.to_numpy(), w)
        codes = dong["dong_code"].to_numpy()
        hh = set(codes[(lm["moran_class"] == "High-High").to_numpy()])
        top = set(codes[mvi.rank(ascending=False, method="min").to_numpy() <= C.TOP_K_VULNERABLE])
        ranks[g["id"]] = pd.Series(mvi.rank(ascending=False, method="min").to_numpy(), index=codes)
        hh_sets[g["id"]], top_sets[g["id"]] = hh, top
        rows.append({**g, "n_HH_significant": len(hh), "HH_dongs": ";".join(sorted(hh)),
                     f"top{C.TOP_K_VULNERABLE}_dongs": ";".join(sorted(top))})

    primary_id = f"{C.MVI_PRIMARY}|decay1.0|product|minmax"
    if primary_id not in ranks:
        raise KeyError(f"primary 시나리오 {primary_id}가 격자에 없음")
    rank_df = pd.DataFrame(ranks)
    grid_df = pd.DataFrame(rows)
    grid_df["spearman_vs_primary"] = [round(float(spearmanr(rank_df[primary_id], rank_df[i])[0]), 4)
                                      for i in grid_df["id"]]
    grid_df["jaccard_HH_vs_primary"] = [round(_jaccard(hh_sets[primary_id], hh_sets[i]), 4) for i in grid_df["id"]]
    grid_df.to_csv(C.TABLES / "robustness_grid.csv", index=False, encoding="utf-8-sig")

    n = len(grid)
    codes = dong["dong_code"].to_numpy()
    freq = pd.DataFrame({
        "dong_code": codes, "dong_name": dong["dong_name"].to_numpy(),
        "hh_share": [sum(c in s for s in hh_sets.values()) / n for c in codes],
        f"top{C.TOP_K_VULNERABLE}_share": [sum(c in s for s in top_sets.values()) / n for c in codes],
        "rank_primary": rank_df[primary_id].astype(int).to_numpy(),
        "rank_mean": rank_df.mean(axis=1).round(2).to_numpy(),
        "rank_sd": rank_df.std(axis=1, ddof=0).round(2).to_numpy(),
        "rank_min": rank_df.min(axis=1).astype(int).to_numpy(),
        "rank_max": rank_df.max(axis=1).astype(int).to_numpy(),
        "hh_in_primary": [c in hh_sets[primary_id] for c in codes],
    })
    freq["hotspot_class"] = np.select(
        [freq["hh_share"] >= C.ROBUST_STABLE_SHARE, freq["hh_share"] >= C.ROBUST_UNSTABLE_SHARE],
        ["stable", "unstable"], default="none")
    freq = freq.sort_values(["hh_share", "rank_mean"], ascending=[False, True])
    freq.to_csv(C.TABLES / "hotspot_frequency.csv", index=False, encoding="utf-8-sig")

    pair_j = [_jaccard(hh_sets[a], hh_sets[b]) for a, b in combinations(hh_sets, 2)]
    pair_rho = [float(spearmanr(rank_df[a], rank_df[b])[0]) for a, b in combinations(ranks, 2)]
    stable = set(freq.loc[freq["hotspot_class"] == "stable", "dong_code"])
    unstable = set(freq.loc[freq["hotspot_class"] == "unstable", "dong_code"])
    name = dict(zip(codes, dong["dong_name"]))
    prim = hh_sets[primary_id]
    out = {
        "grid": {"n_scenarios": n, "mvi_weights": list(C.MVI_SCENARIOS), "decay_scale": C.DECAY_SCALE_VARIANTS,
                 "forms": C.ROBUST_FORMS, "normalizations": C.ROBUST_NORMALIZATIONS,
                 "moran_permutations": C.MORAN_PERMUTATIONS, "seed": C.MORAN_SEED},
        "primary_scenario": primary_id,
        "rank_correlation": {"vs_primary_min": float(grid_df["spearman_vs_primary"].min()),
                             "vs_primary_median": float(grid_df["spearman_vs_primary"].median()),
                             "pairwise_min": round(min(pair_rho), 4),
                             "pairwise_mean": round(float(np.mean(pair_rho)), 4)},
        "jaccard_HH": {"vs_primary_min": float(grid_df["jaccard_HH_vs_primary"].min()),
                       "vs_primary_mean": round(float(grid_df["jaccard_HH_vs_primary"].mean()), 4),
                       "pairwise_mean": round(float(np.mean(pair_j)), 4)},
        "thresholds": {"stable_share": C.ROBUST_STABLE_SHARE, "unstable_share": C.ROBUST_UNSTABLE_SHARE},
        "stable_hotspots": sorted(name[c] for c in stable),
        "unstable_hotspots": sorted(name[c] for c in unstable),
        "no_robustness_vs_robust": {
            "primary_only_HH": sorted(name[c] for c in prim),
            "primary_HH_not_stable": sorted(name[c] for c in prim - stable),
            "stable_not_in_primary_HH": sorted(name[c] for c in stable - prim),
            "note": "primary 한 시나리오만 보면 hotspot으로 보이지만 파라미터에 따라 사라지는 동을 구분한다.",
        },
    }
    (C.TABLES / "robustness_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    print(f"robustness: {n} scenarios, rho_vs_primary_min={out['rank_correlation']['vs_primary_min']}, "
          f"stable={len(stable)}, unstable={len(unstable)}")
    return out


if __name__ == "__main__":
    run_sensitivity()
