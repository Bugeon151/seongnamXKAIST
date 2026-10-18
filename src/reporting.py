"""발표용 결과(outputs/presentation) + 실행 메타데이터(provenance) + 산출물 매니페스트 검사.

- 12_key_numbers.json의 모든 수치는 값·단위·원천 파일·필드 경로를 함께 기록한다
  (발표 자료 숫자는 이 파일에서만 가져온다 — 코드 결과와 발표 숫자 불일치 방지).
- run_metadata.json: 생성 시각, git SHA, Python·패키지 버전, 파라미터, seed, 원천 파일 SHA-256·기준일.
- check_manifest(): 이번 실행에서 생성·갱신되어야 하는 파일이 실제로 새로 만들어졌는지 확인
  (기존 파일이 남아 있다는 이유로 성공 처리하지 않음).
"""
from __future__ import annotations

import datetime as dt
import importlib.metadata as im
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src import config as C

PRESENTATION = C.OUTPUTS / "presentation"
PACKAGES = ["pandas", "numpy", "geopandas", "shapely", "pyproj", "pyogrio", "scipy", "scikit-learn", "libpysal",
            "esda", "pulp", "matplotlib", "folium", "jsonschema"]
# 실행 결과에 영향을 주는 추적 경로 — 이 중 하나라도 커밋과 다르면 code_modified_since_commit=true
PROVENANCE_PATHS = ["src", "run_pipeline.py", "demo.py", "dashboard", "scripts", "requirements.txt", "data/raw",
                    "data/SHA256SUMS",
                    "data/processed/medical_facilities_geocoded.csv", "data/processed/care_facilities_geocoded.csv",
                    "data/processed/depots_geocoded.csv"]
MANIFEST = [
    "tables/data_validation.csv", "preprocessing_report.md",
    "tables/accessibility_by_dong.csv", "tables/accessibility_method_comparison.csv", "json/accessibility_summary.json",
    "tables/mvi_components.csv", "tables/mvi_ranking.csv", "tables/mvi_ablation.csv", "json/mvi_summary.json",
    "tables/local_moran_results.csv", "json/spatial_summary.json", "tables/robustness_grid.csv",
    "tables/hotspot_frequency.csv", "tables/robustness_report.json", "tables/robust_hotspot_meta.json",
    "json/mclp_results.json", "tables/mclp_coverage_curve.csv", "tables/baseline_comparison.csv",
    "json/baseline_summary.json", "tables/policy_comparison.csv", "json/policy_results.json",
    "tables/policy_C_pareto.csv", "json/policy_C_pareto.json", "tables/before_after_by_dong.csv",
    "tables/before_after_by_gu.csv", "json/dong_results.json", "json/explanations.json",
    "json/grounding_validation.json", "json/grounding_failure_demo.json",
    "maps/mvi_map.png", "maps/local_moran_map.png", "maps/robust_hotspot_map.png", "maps/mclp_map.png",
    "maps/policy_routes_map.png", "maps/mvi_interactive.html", "figures/baseline_comparison.png",
    "figures/policy_tradeoff.png", "figures/policy_before_after.png", "figures/policy_kpi_table.png",
]


def _sha256(p: Path) -> str:
    from src.data_validation import sha256_file  # data/SHA256SUMS와 같은 정의(줄바꿈 정규화)
    return sha256_file(p)


def _git() -> dict:
    """코드 버전 기록. git_sha는 '이 산출물을 만든 코드 커밋'이다.

    릴리스 절차상 검증된 코드 커밋에서 실행한 뒤 산출물만 담은 보고 커밋을 그 위에 쌓으므로,
    최종 HEAD가 보고 커밋이면 git_sha는 그 부모(코드 커밋)를 가리키는 것이 정상이다(README '재현 방법').
    """
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=C.BASE_DIR, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", check=True).stdout.rstrip()
    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != C.BASE_DIR.resolve():
            raise subprocess.CalledProcessError(1, "git")  # 다른 저장소 안에 압축해제된 제출본 — 그 SHA를 쓰지 않음
        sha = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--", *PROVENANCE_PATHS)
        return {"git_sha": sha, "code_modified_since_commit": bool(dirty),
                "modified_paths": [ln[3:] for ln in dirty.splitlines()][:20]}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"git_sha": None, "note": "git 저장소가 아님(예: 제출 ZIP) — 코드 버전은 method_version으로 식별"}


def _source_files() -> list[dict]:
    files = [C.RAW_POPULATION, C.RAW_BUS, C.RAW_MEDICAL, C.RAW_CARE, C.RAW_AREA, C.RAW_BOUNDARY,
             *sorted(C.OSM_DIR.glob("*")), C.MEDICAL_GEOCODED, C.CARE_GEOCODED]
    if C.DEPOT_GEOCODED.exists():
        files.append(C.DEPOT_GEOCODED)
    out = []
    for p in files:
        m = re.search(r"_(\d{8})\.csv$", p.name)
        out.append({"path": p.relative_to(C.BASE_DIR).as_posix(), "sha256": _sha256(p), "bytes": p.stat().st_size,
                    "reference_date": f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}" if m else None})
    return out


def write_run_metadata(stage_times: list[dict], started_at: str) -> dict:
    params = {k: getattr(C, k) for k in dir(C) if k.isupper() and isinstance(getattr(C, k), (int, float, str, list, dict, tuple))
              and not isinstance(getattr(C, k), Path)}
    meta = {"project": "MobiLens Seongnam", "method_version": C.METHOD_VERSION, "started_at": started_at,
            "finished_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), **_git(),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "packages": {p: im.version(p) for p in PACKAGES},
            "seeds": {"MORAN_SEED": C.MORAN_SEED, "RANDOM_SEED": C.RANDOM_SEED},
            "parameters": params, "source_files": _source_files(),
            "source_files_hash": "SHA-256 of content with CRLF→LF normalized (data/SHA256SUMS와 동일 정의)",
            "osm_snapshot": json.loads((C.OSM_DIR / "osm_snapshot_meta.json").read_text(encoding="utf-8")),
            "stages": stage_times}
    (C.OUTPUTS / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                                                 encoding="utf-8")
    return meta


def _j(rel: str) -> dict:
    return json.loads((C.OUTPUTS / rel).read_text(encoding="utf-8"))


def build_presentation() -> dict:
    PRESENTATION.mkdir(parents=True, exist_ok=True)
    feat = pd.read_csv(C.DONG_FEATURES, encoding="utf-8-sig")
    comp = pd.read_csv(C.TABLES / "mvi_components.csv", encoding="utf-8-sig", dtype={"dong_code": str})
    acc_s = _j("json/accessibility_summary.json")
    rob = _j("tables/robustness_report.json")
    hot = _j("tables/robust_hotspot_meta.json")
    pol = _j("json/policy_results.json")
    base = _j("json/baseline_summary.json")
    grd = _j("json/grounding_validation.json")
    demo = _j("json/grounding_failure_demo.json")
    geo_m = pd.read_csv(C.MEDICAL_GEOCODED, encoding="utf-8-sig")
    geo_c = pd.read_csv(C.CARE_GEOCODED, encoding="utf-8-sig")
    acc = pd.read_csv(C.TABLES / "accessibility_by_dong.csv", encoding="utf-8-sig")
    top = comp.nsmallest(1, "mvi_rank").iloc[0]
    k = str(C.K_MAIN)

    problem = {
        "n_dong": int(len(feat)), "total_population": int(feat["total_population"].sum()),
        "elderly_population": int(feat["elderly_population"].sum()),
        "elderly_share_pct": round(float(feat["elderly_population"].sum() / feat["total_population"].sum() * 100), 2),
        "medical_facilities_geocoded_in_seongnam": int((geo_m["in_seongnam"].astype(str) == "True").sum()),
        "medical_facilities_total": int(len(geo_m)),
        "welfare_daycare_facilities_used": acc_s["supply"]["n_welfare_used"],
        "welfare_geocoded_in_seongnam": int((geo_c["in_seongnam"].astype(str) == "True").sum()),
        "welfare_total": int(len(geo_c)),
        "dongs_without_welfare_seat_in_catchment": int((acc["wel_capacity_in_catchment"] == 0).sum()),
        "dongs_with_zero_medical_access": int((acc["med_e2sfca_network"] == 0).sum()),
        "dongs_zero_medical_names": acc.loc[acc["med_e2sfca_network"] == 0, "dong_name"].tolist(),
        "medical_e2sfca_p90_over_p10": round(float(acc["med_e2sfca_network"].quantile(0.9)
                                                  / acc["med_e2sfca_network"].quantile(0.1)), 1)
        if acc["med_e2sfca_network"].quantile(0.1) > 0 else None,
        "stable_robust_hotspots": hot["n_robust"], "base_month": C.BASE_MONTH,
    }
    (PRESENTATION / "01_problem_summary.json").write_text(json.dumps(problem, ensure_ascii=False, indent=2),
                                                          encoding="utf-8")
    comp.nsmallest(C.TOP_K_VULNERABLE, "mvi_rank")[
        ["mvi_rank", "dong_name", "mvi", "need", "access", "access_medical", "access_welfare", "access_transit",
         "weakest_access_label", "elderly_population", "elderly_ratio"]].to_csv(
        PRESENTATION / "02_top_vulnerable_regions.csv", index=False, encoding="utf-8-sig")
    copies = {"03_mvi_map.png": C.MAPS / "mvi_map.png", "04_hotspot_map.png": C.MAPS / "robust_hotspot_map.png",
              "05_mclp_map.png": C.MAPS / "policy_routes_map.png",
              "06_policy_comparison.csv": C.TABLES / "policy_comparison.csv",
              "07_policy_tradeoff.png": C.FIGURES / "policy_tradeoff.png",
              "08_before_after.png": C.FIGURES / "policy_before_after.png",
              "09_baseline_comparison.png": C.FIGURES / "baseline_comparison.png"}
    for dst, src in copies.items():
        shutil.copyfile(src, PRESENTATION / dst)
    (PRESENTATION / "10_robustness_summary.json").write_text(json.dumps(
        {k2: rob[k2] for k2 in ["grid", "rank_correlation", "jaccard_HH", "stable_hotspots", "unstable_hotspots",
                                "no_robustness_vs_robust"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    (PRESENTATION / "11_grounding_validation.json").write_text(json.dumps(
        {"summary": grd["summary"], "failure_demo": {kk: v for kk, v in demo.items() if kk != "tampered_text"}},
        ensure_ascii=False, indent=2), encoding="utf-8")

    P = pol["policies"]

    def num(value, unit, src, field, desc):
        return {"value": value, "unit": unit, "source": src, "field": field, "description": desc}

    key = {
        "elderly_population": num(problem["elderly_population"], "명", "data/processed/dong_features_base.csv",
                                  "sum(elderly_population)", f"성남시 65세 이상 인구 ({C.BASE_MONTH})"),
        "top_vulnerable_dong": num(top["dong_name"], "", "outputs/tables/mvi_components.csv", "mvi_rank==1",
                                   "MVI 1위 행정동"),
        "top_vulnerable_mvi": num(float(top["mvi"]), "0~1", "outputs/tables/mvi_components.csv", "mvi (rank 1)",
                                  "MVI 1위 값"),
        "stable_hotspots": num(hot["n_robust"], "개 동", "outputs/tables/robust_hotspot_meta.json", "n_robust",
                               f"강건성 격자 {rob['grid']['n_scenarios']}개 시나리오 중 {C.ROBUST_STABLE_SHARE:.0%} 이상에서 HH·유의"),
        "robust_rank_rho_min": num(rob["rank_correlation"]["vs_primary_min"], "Spearman ρ",
                                   "outputs/tables/robustness_report.json", "rank_correlation.vs_primary_min",
                                   "모든 격자 시나리오와 primary 순위의 최소 상관"),
        "nearest_vs_e2sfca_rho": num(acc_s["method_comparison"]["spearman_nearest_vs_e2sfca"], "Spearman ρ",
                                     "outputs/json/accessibility_summary.json",
                                     "method_comparison.spearman_nearest_vs_e2sfca", "최근접거리 vs E2SFCA 순위 상관"),
        f"mclp_k{k}_vuln_coverage": num(base["by_k"][k]["MCLP"], "비율", "outputs/json/baseline_summary.json",
                                        f"by_k.{k}.MCLP", f"MCLP k={k} 취약수요 커버"),
        f"mclp_minus_top_demand_pp_k{k}": num(base["mclp_minus_top_demand_pp"][k], "%p",
                                              "outputs/json/baseline_summary.json", f"mclp_minus_top_demand_pp.{k}",
                                              "취약수요 상위 동 우선 배치 대비 개선"),
        f"mclp_minus_greedy_pp_k{k}": num(base["mclp_minus_greedy_pp"][k], "%p", "outputs/json/baseline_summary.json",
                                          f"mclp_minus_greedy_pp.{k}", "탐욕 휴리스틱 대비 개선"),
        **{f"policy_{p}_{m}": num(P[p]["kpis"][m], u, "outputs/json/policy_results.json", f"policies.{p}.kpis.{m}",
                                  f"정책 {p} {d}")
           for p in "ABC" for m, u, d in [("vuln_coverage", "비율", "취약수요 커버"),
                                         ("min_gu_coverage", "비율", "최소 구 커버"),
                                         ("total_vehicle_km", "km", "1회 순환 운행거리"),
                                         ("vuln_coverage_per_10km", "비율/10km", "운행 10km당 취약수요 커버")]},
        "grounding_pass": num(f"{grd['summary']['pass']}/{grd['summary']['n']}", "", "outputs/json/grounding_validation.json",
                              "summary", "최종 설명 grounding 검증 통과"),
        "grounding_failure_demo_pass": num(demo["demo_pass"], "", "outputs/json/grounding_failure_demo.json", "demo_pass",
                                           "숫자·지역·정책·인과·지표 변조 5종 모두 FAIL 탐지"),
    }
    (PRESENTATION / "12_key_numbers.json").write_text(json.dumps(key, ensure_ascii=False, indent=2, default=str),
                                                      encoding="utf-8")
    return key


def check_manifest(started_ts: float) -> list[str]:
    """이번 실행에서 생성·갱신되지 않은(또는 없는) 필수 산출물 목록."""
    bad = []
    for rel in MANIFEST:
        p = C.OUTPUTS / rel
        if not p.exists():
            bad.append(f"없음: outputs/{rel}")
        elif p.stat().st_mtime < started_ts - 1:
            bad.append(f"이번 실행에서 갱신되지 않음: outputs/{rel}")
    return bad
