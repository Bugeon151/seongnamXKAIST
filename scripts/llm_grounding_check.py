"""실제 로컬 LLM(Ollama)으로 설명을 생성해 grounding 검증 통과율을 측정 (선택 실행).

사용: python scripts/llm_grounding_check.py --model gemma3:4b [--n 10]
전제: python run_pipeline.py 실행 완료, Ollama 서버 실행 중(기본 http://localhost:11434)
결과: outputs/json/llm_grounding_check.json — 대상별 시도 텍스트, 위반 사유, 채택/대체 여부.
LLM 출력은 모델·버전에 따라 달라지므로 이 파일은 '측정 기록'이며 핵심 결과 재현 대상이 아니다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _kind(v: str) -> str:
    """위반 사유 문장 → 유형."""
    for key, label in [("KPI", "KPI 값 불일치"), ("입력에 없는 숫자", "입력에 없는 숫자"), ("지역명", "입력에 없는 지역"),
                       ("행정동", "존재하지 않는 행정동"), ("행정코드", "다른 행정코드"), ("정책", "입력에 없는 정책"),
                       ("지표", "입력에 없는 지표"), ("인과", "인과 주장"), ("단정", "금지 단정"), ("누락", "대상 지역명 누락")]:
        if key in v:
            return label
    return "기타"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=10, help="설명할 동 수 (MVI 상위 n개) + 정책 3개")
    a = ap.parse_args()
    os.environ["OLLAMA_MODEL"] = a.model
    from src import config as C
    from src.explanation import explain, build_policy_evidence, ollama_generate
    recs = sorted(json.loads((C.JSON_DIR / "dong_results.json").read_text(encoding="utf-8")),
                  key=lambda r: r["mvi_rank"])[:a.n]
    items = [("dong", r["region_name"], r) for r in recs] + [("policy", r["name"], r) for r in build_policy_evidence()]
    out = []
    for kind, name, rec in items:
        e = explain(rec, ollama_generate)
        out.append({"kind": kind, "target": name, "generator": e["generator"], "llm_error": e["llm_error"],
                    "attempts": e["attempts"], "final_text": e["final_text"]})
        print(f"{kind:6s} {name:14s} → {e['generator']}")
    llm_attempts = [t for o in out for t in o["attempts"] if t["generator"] == "llm"]
    summary = {"model": a.model, "n_targets": len(out), "final_generator": dict(Counter(o["generator"] for o in out)),
               "llm_attempts": len(llm_attempts), "llm_attempts_pass": sum(t["FINAL"] == "PASS" for t in llm_attempts),
               "violation_types": dict(Counter(_kind(v) for t in llm_attempts for v in t["violations"])),
               "note": "최종 채택 설명은 모두 grounding PASS (FAIL이면 재생성 후 템플릿으로 대체)"}
    (C.JSON_DIR / "llm_grounding_check.json").write_text(
        json.dumps({"summary": summary, "items": out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
