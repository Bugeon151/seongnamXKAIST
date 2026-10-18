"""MobiLens Seongnam — 전체 파이프라인 (단일 명령 재현).

사용:
  python run_pipeline.py                      raw → 전처리 → 분석 → outputs/ 전체 재생성
  python run_pipeline.py --validate           outputs_validation/ 에 새로 생성 후 저장소 outputs/와 비교
  python run_pipeline.py --out DIR            산출물을 DIR 아래(data/processed, outputs)에 생성
  python run_pipeline.py --demo [--dong 코드] [--tamper]   생성된 결과로 CLI 시연 (재계산 없음)
  python run_pipeline.py --what-if k=7 policy=B radius_m=1000   파라미터 변경 재계산
  python run_pipeline.py --llm-model gemma3:4b  로컬 Ollama로 설명 생성(선택, grounding 검증 후 채택)

단계: [1/8] 데이터 검증 → [2/8] 전처리 → [3/8] 접근성 → [4/8] MVI → [5/8] 공간분석
      → [6/8] 최적화 → [7/8] 설명 검증 → [8/8] 산출물.  단계가 FAIL하면 즉시 중단(이후 단계 SKIPPED).
지오코딩(외부 API)은 파이프라인에서 재실행하지 않고, 버전관리되는 결과 파일을 검증 후 사용한다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="MobiLens Seongnam pipeline")
    ap.add_argument("--out", help="산출물 루트 (기본: 저장소 루트)")
    ap.add_argument("--validate", action="store_true", help="outputs_validation/에 재생성 후 기존 결과와 비교")
    ap.add_argument("--demo", action="store_true", help="생성된 결과로 CLI 시연")
    ap.add_argument("--dong", help="--demo에서 설명할 행정동 코드")
    ap.add_argument("--tamper", action="store_true", help="--demo에서 고의 오류 검증 시연")
    ap.add_argument("--what-if", nargs="*", metavar="KEY=VALUE", help="What-if 재계산 파라미터")
    ap.add_argument("--llm-model", help="Ollama 모델명 (설정 시 LLM 설명 시도)")
    return ap.parse_args(argv)


def run_stages() -> int:
    from src import config as C
    from src.data_validation import validate_raw, validate_processed
    from src.preprocessing import run_preprocessing
    from src.accessibility import run_accessibility
    from src.mvi import run_mvi
    from src.spatial_statistics import run_spatial_stats
    from src.sensitivity import run_sensitivity
    from src.hotspot import run_hotspot
    from src.optimization import run_optimization
    from src.baseline import run_baseline
    from src.policy_scenarios import run_policies
    from src.explanation import build_structured_results, generate_explanations
    from src.grounding_validator import run_validation as run_grounding, intentional_failure_demo
    from src.visualization import run_visualization
    from src.reporting import build_presentation, write_run_metadata, check_manifest
    import pandas as pd

    for d in [C.MAPS, C.TABLES, C.JSON_DIR, C.FIGURES, C.LOGS, C.PROCESSED_DIR, C.INTERIM_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    started_ts, started_at = time.time(), dt.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"MobiLens Seongnam {C.METHOD_VERSION} | output root: {C.OUT_ROOT}")
    print(f"LLM: {'Ollama ' + os.environ['OLLAMA_MODEL'] if os.environ.get('OLLAMA_MODEL') else '미사용(템플릿 설명)'}")

    def _outputs():
        run_visualization()
        build_presentation()
        bad = check_manifest(started_ts)
        if bad:
            raise RuntimeError("필수 산출물 누락/미갱신: " + "; ".join(bad))

    stages = [
        ("Data validation", [validate_raw]),
        ("Preprocessing", [run_preprocessing, validate_processed]),
        ("Accessibility", [run_accessibility]),
        ("MVI", [run_mvi]),
        ("Spatial analysis", [run_spatial_stats, run_sensitivity, run_hotspot]),
        ("Optimization", [run_optimization, run_baseline, run_policies]),
        ("Explanation validation", [build_structured_results, generate_explanations, run_grounding,
                                    intentional_failure_demo]),
        ("Outputs", [_outputs]),
    ]
    status: list[dict] = []
    failed = False
    for i, (name, fns) in enumerate(stages, 1):
        label = f"[{i}/{len(stages)}] {name}"
        if failed:
            status.append({"stage": label, "status": "SKIPPED", "seconds": 0.0, "detail": "선행 단계 FAIL"})
            print(f"{label:34s} SKIPPED")
            continue
        print(f"\n{label} ...", flush=True)
        t0 = time.time()
        try:
            for fn in fns:
                fn()
            sec = round(time.time() - t0, 1)
            status.append({"stage": label, "status": "PASS", "seconds": sec, "detail": ""})
            print(f"{label:34s} PASS ({sec}s)")
        except Exception as e:  # noqa: BLE001 — 단계 경계에서 원인을 기록하고 중단
            sec = round(time.time() - t0, 1)
            status.append({"stage": label, "status": "FAIL", "seconds": sec, "detail": f"{type(e).__name__}: {e}"})
            print(f"{label:34s} FAIL ({sec}s) — {type(e).__name__}: {e}")
            traceback.print_exc(limit=4)
            failed = True
    pd.DataFrame(status).to_csv(C.LOGS / "pipeline_status.csv", index=False, encoding="utf-8-sig")
    if not failed:
        write_run_metadata(status, started_at)
    print("\n===== PIPELINE SUMMARY =====")
    for s in status:
        print(f"{s['stage']:34s} {s['status']:8s} {s['seconds']:>6}s {s['detail']}")
    total = round(sum(s["seconds"] for s in status), 1)
    n_pass = sum(s["status"] == "PASS" for s in status)
    print(f"RESULT: {n_pass}/{len(status)} stages passed, total {total}s")
    return 1 if failed else 0


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.validate and not args.out:
        args.out = "outputs_validation"
    if args.out:
        os.environ["MOBILENS_OUT_ROOT"] = args.out
    if args.llm_model:
        os.environ["OLLAMA_MODEL"] = args.llm_model
    if args.demo:
        from demo import run_demo
        return run_demo(args.dong, args.tamper)
    if args.what_if is not None:
        from demo import run_what_if_cli
        return run_what_if_cli(args.what_if)
    code = run_stages()
    if code == 0 and args.validate:
        sys.path.insert(0, str(BASE_DIR / "scripts"))
        from compare_outputs import main as compare_main
        compare_main([args.out])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
