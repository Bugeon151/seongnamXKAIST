"""MVI (Mobility Vulnerability Index) — Need-Access mismatch 지수.

Need_i   = mean( norm(65세 이상 인구), norm(고령인구 비율) )                ∈ [0,1]
Access_i = mean( norm(의료 E2SFCA), norm(복지 E2SFCA), norm(고령 1천명당 정류장) ) ∈ [0,1]
MVI_i    = Need_i^(2·w_need) × (1 − Access_i)^(2·w_access)                  ∈ [0,1]
           균형 시나리오(w=0.5/0.5)에서 MVI = Need × (1 − Access)  (예선 정의)
norm: min-max (primary) / 순위 백분위(강건성 분석용, 이상치 영향 완화)
  min-max에서 모든 값이 같으면 0.5 (정보 없음)로 둔다.
곱 형식이라 Need 또는 Access 결핍 중 하나가 0이면 MVI=0 — '수요와 결핍이 동시에 큰 곳'을 찾는 설계.

출력:
  data/processed/mvi_results.csv          시나리오별 MVI·순위 (wide)
  outputs/tables/mvi_components.csv       동별 원인 분해 (원자료 → 정규화 구성요소 → MVI, 주된 결핍 요인)
  outputs/tables/mvi_ranking.csv          primary 시나리오 순위표
  outputs/tables/mvi_ablation.csv         구성요소 제거·결합형식 변경 시 순위 변화
  outputs/tables/feature_correlation.csv  입력 지표 상관 (double counting 점검)
  outputs/json/mvi_summary.json           상위 지역 원인 추적·이상치 점검
"""
from __future__ import annotations

import json

import pandas as pd
from scipy.stats import spearmanr

from src import config as C

NEED_COLS = ["elderly_population", "elderly_ratio"]
COMPONENT_LABEL = {"medical": "의료 접근성", "welfare": "복지(주야간보호) 접근성", "transit": "대중교통 접근성"}


def minmax(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def rank_norm(s: pd.Series) -> pd.Series:
    """순위 백분위 정규화 (평균순위, 0=최소 ~ 1=최대)."""
    if s.nunique() == 1:
        return pd.Series(0.5, index=s.index)
    return (s.rank(method="average") - 1) / (len(s) - 1)


def normalize(s: pd.Series, method: str = "minmax") -> pd.Series:
    if method == "minmax":
        return minmax(s)
    if method == "rank":
        return rank_norm(s)
    raise ValueError(f"알 수 없는 정규화: {method}")


def compute_components(df: pd.DataFrame, normalization: str = "minmax",
                       access_components: dict[str, str] | None = None) -> pd.DataFrame:
    """동별 Need·Access 구성요소 (0~1). df에는 NEED_COLS와 접근성 컬럼이 있어야 한다."""
    comps = access_components or C.ACCESS_COMPONENTS
    out = pd.DataFrame(index=df.index)
    for c in NEED_COLS:
        out[f"need_{c}"] = normalize(df[c], normalization)
    out["need"] = out[[f"need_{c}" for c in NEED_COLS]].mean(axis=1)
    for name, col in comps.items():
        out[f"access_{name}"] = normalize(df[col], normalization)
    out["access"] = out[[f"access_{n}" for n in comps]].mean(axis=1)
    out["access_deficit"] = 1.0 - out["access"]
    if out.isna().any().any():
        raise ValueError(f"MVI 구성요소 결측: {out.columns[out.isna().any()].tolist()}")
    if ((out < -1e-12) | (out > 1 + 1e-12)).any().any():
        raise ValueError("MVI 구성요소 정규화 범위 [0,1] 위반")
    return out


def combine(need: pd.Series, deficit: pd.Series, w_need: float, w_access: float,
            form: str = "product") -> pd.Series:
    """Need와 접근성 결핍 결합. product: Need^(2w_n)·Deficit^(2w_a), additive: 가중합."""
    if abs(w_need + w_access - 1.0) > 1e-9:
        raise ValueError("w_need + w_access = 1 이어야 함")
    if form == "product":
        return need.pow(2 * w_need) * deficit.pow(2 * w_access)
    if form == "additive":
        return w_need * need + w_access * deficit
    raise ValueError(f"알 수 없는 결합형식: {form}")


def load_mvi_inputs() -> pd.DataFrame:
    feat = pd.read_csv(C.DONG_FEATURES, encoding="utf-8-sig", dtype={"dong_code": str})
    acc = pd.read_csv(C.PROCESSED_DIR / "accessibility.csv", encoding="utf-8-sig", dtype={"dong_code": str})
    df = feat.merge(acc.drop(columns=["dong_name", "elderly_population", "bus_stop_count"]),
                    on="dong_code", how="inner", validate="one_to_one")
    if len(df) != C.N_DONG:
        raise ValueError(f"특성-접근성 결합 후 {len(df)}개 동 (기대 {C.N_DONG})")
    return df


def _ablation(df: pd.DataFrame, primary: pd.Series) -> pd.DataFrame:
    w_n, w_a = C.MVI_SCENARIOS[C.MVI_PRIMARY]
    top = set(primary.nlargest(C.TOP_K_VULNERABLE).index)
    variants: dict[str, pd.Series] = {}
    for drop in C.ACCESS_COMPONENTS:
        keep = {k: v for k, v in C.ACCESS_COMPONENTS.items() if k != drop}
        comp = compute_components(df, access_components=keep)
        variants[f"without_{drop}"] = combine(comp["need"], comp["access_deficit"], w_n, w_a)
    comp = compute_components(df)
    variants["need_only"] = comp["need"]
    variants["access_deficit_only"] = comp["access_deficit"]
    variants["additive_form"] = combine(comp["need"], comp["access_deficit"], w_n, w_a, "additive")
    rows = []
    for name, v in variants.items():
        rho, _ = spearmanr(primary, v)
        t = set(v.nlargest(C.TOP_K_VULNERABLE).index)
        rows.append({"variant": name, "spearman_vs_primary": round(float(rho), 4),
                     f"top{C.TOP_K_VULNERABLE}_overlap": len(top & t),
                     "jaccard_top": round(len(top & t) / len(top | t), 4),
                     "top_dongs": ", ".join(df.loc[sorted(t, key=lambda i: -v[i]), "dong_name"])})
    return pd.DataFrame(rows)


def run_mvi() -> pd.DataFrame:
    df = load_mvi_inputs()
    comp = compute_components(df)

    corr_cols = NEED_COLS + ["elderly_density"] + list(C.ACCESS_COMPONENTS.values())
    df[corr_cols].corr(method="pearson").round(3).to_csv(C.TABLES / "feature_correlation.csv",
                                                          encoding="utf-8-sig")
    rows = []
    for scen, (w_need, w_acc) in C.MVI_SCENARIOS.items():
        mvi = combine(comp["need"], comp["access_deficit"], w_need, w_acc)
        r = df[["dong_code", "dong_name"]].copy()
        r["scenario"], r["w_need"], r["w_access"] = scen, w_need, w_acc
        r["need_norm"] = comp["need"].round(4)
        r["access_norm"] = comp["access"].round(4)
        r["mvi"] = mvi.round(4)
        r["mvi_rank"] = mvi.rank(ascending=False, method="min").astype(int)
        rows.append(r)
    long_df = pd.concat(rows, ignore_index=True)
    if not long_df["mvi"].between(0, 1).all():
        raise ValueError("MVI 범위 [0,1] 위반")

    wide = long_df.pivot(index=["dong_code", "dong_name"], columns="scenario", values=["mvi", "mvi_rank"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(C.PROCESSED_DIR / "mvi_results.csv", index=False, encoding="utf-8-sig")
    long_df[long_df["scenario"] == C.MVI_PRIMARY].sort_values("mvi_rank").to_csv(
        C.TABLES / "mvi_ranking.csv", index=False, encoding="utf-8-sig")

    # 원인 분해: 원자료 → 정규화 구성요소 → MVI
    w_n, w_a = C.MVI_SCENARIOS[C.MVI_PRIMARY]
    primary = combine(comp["need"], comp["access_deficit"], w_n, w_a)
    dec = df[["dong_code", "dong_name", "elderly_population", "elderly_ratio"]
             + list(C.ACCESS_COMPONENTS.values())].copy()
    dec = pd.concat([dec, comp.round(4)], axis=1)
    dec["mvi"] = primary.round(4)
    dec["mvi_rank"] = primary.rank(ascending=False, method="min").astype(int)
    acc_cols = [f"access_{n}" for n in C.ACCESS_COMPONENTS]
    dec["weakest_access"] = comp[acc_cols].idxmin(axis=1).str.replace("access_", "", regex=False)
    dec["weakest_access_label"] = dec["weakest_access"].map(COMPONENT_LABEL)
    dec["need_rank"] = comp["need"].rank(ascending=False, method="min").astype(int)
    dec["deficit_rank"] = comp["access_deficit"].rank(ascending=False, method="min").astype(int)
    dec = dec.sort_values("mvi_rank")
    dec.to_csv(C.TABLES / "mvi_components.csv", index=False, encoding="utf-8-sig")

    abl = _ablation(df, primary)
    abl.to_csv(C.TABLES / "mvi_ablation.csv", index=False, encoding="utf-8-sig")

    raw_cols = NEED_COLS + list(C.ACCESS_COMPONENTS.values())
    z = (df[raw_cols] - df[raw_cols].mean()) / df[raw_cols].std(ddof=0)
    outliers = {c: df.loc[z[c].abs() > 3, "dong_name"].tolist() for c in raw_cols if (z[c].abs() > 3).any()}
    top = dec.head(C.TOP_K_VULNERABLE)
    summary = {
        "formula": "MVI = Need^(2·w_need) × (1 − Access)^(2·w_access); primary(0.5,0.5) = Need × (1 − Access)",
        "need": "mean(minmax(65세 이상 인구), minmax(고령인구 비율))",
        "access": "mean(minmax(의료 E2SFCA 네트워크), minmax(복지 E2SFCA 네트워크), minmax(고령 1천명당 정류장))",
        "primary_scenario": C.MVI_PRIMARY,
        "range_check": {"min": float(primary.min()), "max": float(primary.max())},
        "outliers_abs_z_gt_3": outliers,
        "outlier_note": "이상치는 제거하지 않는다. 순위 정규화 시나리오(강건성 격자)로 영향 점검.",
        "top_regions_trace": [
            {"rank": int(r.mvi_rank), "dong_name": r.dong_name, "mvi": float(r.mvi),
             "need": float(r.need), "need_rank": int(r.need_rank),
             "access": float(r.access), "deficit_rank": int(r.deficit_rank),
             "access_medical": float(r.access_medical), "access_welfare": float(r.access_welfare),
             "access_transit": float(r.access_transit), "weakest_access": r.weakest_access_label,
             "elderly_population": int(r.elderly_population), "elderly_ratio": float(r.elderly_ratio)}
            for r in top.itertuples()],
        "ablation": abl.to_dict("records"),
    }
    (C.JSON_DIR / "mvi_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    print(f"mvi: scenarios={list(C.MVI_SCENARIOS)}, top3="
          f"{dec.head(3)[['dong_name', 'mvi']].to_dict('records')}")
    return long_df


if __name__ == "__main__":
    run_mvi()
