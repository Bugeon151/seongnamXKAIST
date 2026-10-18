"""MobiLens CLI 데모 — 파이프라인이 생성한 결과를 읽어 핵심 흐름을 보여준다 (분석 재계산 없음).

실행: python demo.py [동코드] [--tamper]      (또는 python run_pipeline.py --demo)
흐름: 문제 요약 → 취약 상위 동 → 선택 동 설명·근거·검증 → (고의 오류 FAIL) → 정책 A/B/C 비교
      → What-if 1회 실제 재계산(k 변경)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def _load(rel: str):
    from src import config as C
    p = C.OUTPUTS / rel
    if not p.exists():
        raise FileNotFoundError(f"{p} 없음 — 먼저 python run_pipeline.py 를 실행하세요")
    return json.loads(p.read_text(encoding="utf-8"))


def run_demo(dong_code: str | None = None, tamper: bool = False) -> int:
    from src import config as C
    from src.grounding_validator import validate_explanation, intentional_failure_demo
    meta = _load("run_metadata.json")
    print(f"=== MobiLens Seongnam 데모 (결과 생성: {meta['finished_at']}, git {str(meta.get('git_sha'))[:8]}) ===")
    prob = _load("presentation/01_problem_summary.json")
    print(f"\n[문제] 성남시 {prob['n_dong']}개 행정동, 65세 이상 {prob['elderly_population']:,}명"
          f"({prob['elderly_share_pct']}%). 복지(주야간보호) 캐치먼트에 정원이 없는 동 "
          f"{prob['dongs_without_welfare_seat_in_catchment']}개, 안정 Robust Hotspot {prob['stable_robust_hotspots']}개.")
    recs = _load("json/dong_results.json")
    top = sorted(recs, key=lambda r: r["mvi_rank"])[:5]
    print("\n[취약 상위 5개 동] " + ", ".join(f"{r['mvi_rank']}. {r['region_name']}({r['mvi']})" for r in top))
    exps = {e["id"]: e for e in _load("json/explanations.json")}
    code = dong_code or top[0]["region_id"]
    if code not in exps:
        raise KeyError(f"행정동 코드 {code} 없음")
    e = exps[code]
    print(f"\n[근거 JSON] {json.dumps({k: e['evidence'][k] for k in ['region_name', 'mvi', 'mvi_rank', 'need', 'access', 'weakest_access', 'hotspot_class']}, ensure_ascii=False)}")
    print(f"[설명 — {e['generator']}] {e['final_text']}")
    v = validate_explanation(e["final_text"], e["evidence"])
    print(f"[grounding 검증] FINAL={v['FINAL']} (숫자 {len(v['numeric'])}개·지역 {v['mentioned_places']}·정책 {v['policy_mentions']})")
    if tamper:
        d = intentional_failure_demo()
        for k, r in d.items():
            if isinstance(r, dict) and "FINAL" in r:
                print(f"  - {k:16s} → {r['FINAL']}  {r['violations'][:1]}")
    pol = _load("json/policy_results.json")
    print("\n[정책 A/B/C — 동일 KPI]")
    for k, p in pol["policies"].items():
        kp = p["kpis"]
        print(f"  {p['name']:12s} 거점 {kp['n_new_stops']:>2}개 | 취약커버 {kp['vuln_coverage']:.3f} | 최소구 "
              f"{kp['min_gu_coverage']:.3f} | 운행 {kp['total_vehicle_km']:.1f}km | 10km당 {kp['vuln_coverage_per_10km']:.3f}")
    print(f"  → {pol['tradeoff']}")
    print("\n[What-if] 정책 A를 거점 7개로 늘리면 (실제 재계산) ...")
    t = time.time()
    run_what_if_cli(["policy=A", "k=7"], quiet=True)
    print(f"  (재계산 {time.time() - t:.1f}s)")
    del C
    return 0


def run_what_if_cli(kvs: list[str], quiet: bool = False) -> int:
    from dataclasses import fields
    from src.what_if import WhatIfParams, run_what_if
    types = {f.name: f.type for f in fields(WhatIfParams)}
    kw = {}
    for kv in kvs:
        k, _, v = kv.partition("=")
        if k not in types:
            raise KeyError(f"알 수 없는 파라미터 {k} (가능: {list(types)})")
        kw[k] = int(v) if types[k] in (int, "int") else float(v) if types[k] in (float, "float") else v
    r = run_what_if(WhatIfParams(**kw))
    if "result" not in r:
        print(f"  {r['message']} (status={r['status']})")
        return 1
    kp = r["result"]["kpis"]
    print(f"  params={ {k: v for k, v in r['params'].items() if k in kw} } status={r['status']}")
    print(f"  거점 {kp['n_new_stops']}개 | 취약커버 {kp['vuln_coverage']:.3f} | 최소구 {kp['min_gu_coverage']:.3f} | "
          f"운행 {kp['total_vehicle_km']:.1f}km | 차량 {kp['vehicles_used']}대 | 최대순환 {kp['max_cycle_min']:.0f}분")
    if not quiet:
        print("  선정 거점: " + ", ".join(f"{s['name']}({s['dong']})" for s in r["result"]["selected_stops"]))
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(run_demo(args[0] if args else None, "--tamper" in sys.argv))
