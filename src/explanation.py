"""근거제약형 설명 — LLM은 계산하지 않고, 분석엔진이 만든 구조화 JSON(evidence)만 문장으로 풀어쓴다.

흐름: evidence 생성 → JSON Schema 검증 → (선택) LLM 생성 → 파싱 → grounding 검증
      → PASS면 채택 / FAIL이면 위반 내용을 알려주고 LLM_MAX_RETRIES회 재생성
      → 그래도 FAIL이면 결정적 템플릿 설명으로 대체 (모든 시도와 사유를 기록)
LLM은 선택 계층이다: 환경변수 OLLAMA_MODEL이 없으면 템플릿만 사용하고, 설정되었는데 서버에
연결할 수 없으면 오류를 기록·경고한 뒤 템플릿을 사용한다(분석·최적화에는 영향 없음).

출력: outputs/json/dong_results.json (동별 evidence), outputs/json/explanations.json,
      outputs/json/evidence_schema.json
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

import pandas as pd
from jsonschema import validate as js_validate

from src import config as C
from src.grounding_validator import validate_explanation
from src.routing import dong_gu_map

_NUM = {"type": "number"}
_BOOL = {"type": "boolean"}
DONG_SCHEMA = {
    "type": "object",
    "required": ["kind", "region_id", "region_name", "gu", "legal_dongs", "elderly_population", "elderly_ratio",
                 "need", "access", "access_medical", "access_welfare", "access_transit", "weakest_access",
                 "mvi", "mvi_rank", "local_moran_class", "local_moran_p", "hotspot_class", "hotspot_share",
                 "n_robust_scenarios", "covered_by_policy", "policies_available", "method_version"],
    "properties": {
        "kind": {"const": "dong"}, "region_id": {"type": "string", "pattern": r"^\d{10}$"},
        "region_name": {"type": "string"}, "gu": {"enum": ["수정구", "중원구", "분당구"]},
        "legal_dongs": {"type": "array", "items": {"type": "string"}},
        "elderly_population": {"type": "integer", "minimum": 0}, "elderly_ratio": _NUM,
        "need": {"type": "number", "minimum": 0, "maximum": 1}, "access": {"type": "number", "minimum": 0, "maximum": 1},
        "access_medical": _NUM, "access_welfare": _NUM, "access_transit": _NUM, "weakest_access": {"type": "string"},
        "mvi": {"type": "number", "minimum": 0, "maximum": 1}, "mvi_rank": {"type": "integer", "minimum": 1},
        "local_moran_class": {"enum": ["High-High", "High-Low", "Low-High", "Low-Low", "Not Significant"]},
        "local_moran_p": {"type": "number", "minimum": 0, "maximum": 1},
        "hotspot_class": {"enum": ["stable", "unstable", "none"]},
        "hotspot_share": {"type": "number", "minimum": 0, "maximum": 1}, "n_robust_scenarios": {"type": "integer"},
        "covered_by_policy": {"type": "object", "properties": {"A": _BOOL, "B": _BOOL, "C": _BOOL},
                              "required": ["A", "B", "C"]},
        "policies_available": {"type": "array", "items": {"enum": ["A", "B", "C"]}},
        "method_version": {"type": "string"},
    },
}
POLICY_SCHEMA = {
    "type": "object",
    "required": ["kind", "policy", "name", "goal", "kpis", "compare", "allowed_places", "policies_available"],
    "properties": {
        "kind": {"const": "policy"}, "policy": {"enum": ["A", "B", "C"]}, "name": {"type": "string"},
        "goal": {"type": "string"},
        "kpis": {"type": "object", "required": ["n_new_stops", "vuln_coverage", "covered_elderly", "min_gu_coverage",
                                                "total_vehicle_km", "max_cycle_min", "vehicles_used",
                                                "vuln_coverage_per_10km"]},
        "compare": {"type": "object", "required": ["A", "B", "C"]},
        "allowed_places": {"type": "array", "items": {"type": "string"}},
        "policies_available": {"type": "array"},
    },
}
HOTSPOT_LABEL = {"stable": "안정 Robust Hotspot", "unstable": "조건부(불안정) hotspot", "none": "hotspot 아님"}


def build_structured_results() -> list[dict]:
    """동별 evidence 레코드 생성 (분석 산출물에서만 값을 가져온다)."""
    comp = pd.read_csv(C.TABLES / "mvi_components.csv", encoding="utf-8-sig", dtype={"dong_code": str})
    moran = pd.read_csv(C.TABLES / "local_moran_results.csv", encoding="utf-8-sig", dtype={"dong_code": str})
    moran = moran[moran["scenario"] == C.MVI_PRIMARY].set_index("dong_code")
    freq = pd.read_csv(C.TABLES / "hotspot_frequency.csv", encoding="utf-8-sig", dtype={"dong_code": str})
    freq = freq.set_index("dong_code")
    pol = json.loads((C.JSON_DIR / "policy_results.json").read_text(encoding="utf-8"))["policies"]
    grid_n = len(pd.read_csv(C.TABLES / "robustness_grid.csv", encoding="utf-8-sig"))
    area = pd.read_csv(C.RAW_AREA, encoding="utf-8-sig", dtype=str)
    legal = {(r["구청명"].strip(), r["동 명"].strip()): [x.strip() for x in str(r["비고(관할 법정동)"]).split("+")]
             for _, r in area.iterrows()}
    gmap = dong_gu_map()
    recs = []
    for r in comp.sort_values("dong_code").itertuples():
        code, gu = r.dong_code, gmap[r.dong_code]
        rec = {"kind": "dong", "region_id": code, "region_name": r.dong_name, "gu": gu,
               "legal_dongs": legal[(gu, r.dong_name)],
               "elderly_population": int(r.elderly_population), "elderly_ratio": round(float(r.elderly_ratio), 2),
               "need": round(float(r.need), 4), "access": round(float(r.access), 4),
               "access_medical": round(float(r.access_medical), 4),
               "access_welfare": round(float(r.access_welfare), 4),
               "access_transit": round(float(r.access_transit), 4),
               "weakest_access": r.weakest_access_label,
               "mvi": round(float(r.mvi), 4), "mvi_rank": int(r.mvi_rank),
               "local_moran_class": str(moran.loc[code, "moran_class"]),
               "local_moran_p": round(float(moran.loc[code, "moran_p"]), 4),
               "hotspot_class": str(freq.loc[code, "hotspot_class"]),
               "hotspot_share": round(float(freq.loc[code, "hh_share"]), 4), "n_robust_scenarios": grid_n,
               "covered_by_policy": {k: code in v["covered_dongs"] for k, v in pol.items()},
               "policies_available": sorted(pol), "method_version": C.METHOD_VERSION}
        js_validate(rec, DONG_SCHEMA)
        recs.append(rec)
    (C.JSON_DIR / "dong_results.json").write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
    (C.JSON_DIR / "evidence_schema.json").write_text(
        json.dumps({"dong": DONG_SCHEMA, "policy": POLICY_SCHEMA}, ensure_ascii=False, indent=2), encoding="utf-8")
    return recs


def build_policy_evidence() -> list[dict]:
    res = json.loads((C.JSON_DIR / "policy_results.json").read_text(encoding="utf-8"))["policies"]
    keys = ["vuln_coverage", "min_gu_coverage", "total_vehicle_km", "vuln_coverage_per_10km"]
    compare = {k: {c: v["kpis"][c] for c in keys} for k, v in res.items()}
    out = []
    for k, v in res.items():
        kp = v["kpis"]
        places = sorted({s["dong"] for s in v["selected_stops"]} | set(kp["gu_coverages"]))
        rec = {"kind": "policy", "policy": k, "name": v["name"], "goal": v["goal"],
               "kpis": {c: kp[c] for c in ["n_new_stops", "vuln_coverage", "elderly_coverage", "covered_elderly",
                                          "min_gu_coverage", "total_vehicle_km", "max_cycle_min", "vehicles_used",
                                          "vuln_coverage_per_10km"]},
               "compare": compare, "allowed_places": places, "policies_available": sorted(res)}
        js_validate(rec, POLICY_SCHEMA)
        out.append(rec)
    return out


def template_explain(rec: dict) -> str:
    """결정적 템플릿 — evidence 값만 사용한다."""
    if rec["kind"] == "dong":
        cov = ", ".join(f"정책 {k} {'포함' if v else '미포함'}" for k, v in rec["covered_by_policy"].items())
        return (f"{rec['region_name']}({rec['gu']}, 코드 {rec['region_id']})의 65세 이상 인구는 "
                f"{rec['elderly_population']}명, 고령인구 비율 {rec['elderly_ratio']}%이다. "
                f"MVI(수요×접근 결핍)는 {rec['mvi']}로 {C.N_DONG}개 동 중 {rec['mvi_rank']}위이며, "
                f"수요 지수 Need {rec['need']}, 종합 접근성 Access {rec['access']}이다. "
                f"접근성 구성요소 중 가장 낮은 것은 {rec['weakest_access']}이다"
                f"(정규화 값: 의료 {rec['access_medical']}, 복지 {rec['access_welfare']}, "
                f"대중교통 {rec['access_transit']}). "
                f"Local Moran 분류는 {rec['local_moran_class']}(p={rec['local_moran_p']})이고, "
                f"강건성 격자 {rec['n_robust_scenarios']}개 시나리오 중 {round(rec['hotspot_share'] * 100, 1)}%에서 "
                f"고-고 군집으로 나타나 {HOTSPOT_LABEL[rec['hotspot_class']]}에 해당한다. "
                f"신규 거점 커버 여부는 {cov}이다. "
                "본 수치는 공개 집계자료로 계산한 잠재 접근성 지표이며 인과관계를 뜻하지 않는다.")
    k = rec["kpis"]
    cmp_txt = ", ".join(f"정책 {p} {v['vuln_coverage']}" for p, v in rec["compare"].items())
    return (f"정책 {rec['policy']}({rec['name'].split('_', 1)[1]})의 목표는 '{rec['goal']}'이다. "
            f"신규 거점 {k['n_new_stops']}개로 취약 고령수요의 {round(k['vuln_coverage'] * 100, 1)}%"
            f"(고령인구 {k['covered_elderly']}명)를 커버하며, 구별 최소 커버율은 {k['min_gu_coverage']}이다. "
            f"구 보건소 차량 {k['vehicles_used']}대가 1회 순환에 총 {k['total_vehicle_km']}km를 운행하고 "
            f"최대 순환시간은 {k['max_cycle_min']}분이며, 운행 10km당 취약수요 커버율은 "
            f"{k['vuln_coverage_per_10km']}이다. 같은 KPI의 취약수요 커버율은 {cmp_txt}이며, "
            "우열 판단은 의사결정자의 몫이다.")


def _parse_llm(text: str) -> str:
    """LLM 출력 정리: 사고과정 태그·마크다운 제거, 공백 정규화."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"[*#>`]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _prompt(rec: dict, feedback: list[str] | None) -> str:
    rules = ("너는 행정 분석 결과를 설명하는 도우미다. 아래 JSON에 있는 값과 지역명만 사용해 한국어 3~5문장으로 설명하라. "
             "규칙: (1) JSON에 없는 숫자·지역·정책·지표를 만들지 마라. (2) 숫자는 JSON 값 그대로 또는 같은 자리로 반올림해 쓰고, "
             "비율은 백분율로 써도 된다. (3) '때문에', '로 인해', '원인' 같은 인과 표현을 쓰지 마라. "
             "(4) 대상 지역명(region_name)이 있으면 반드시 포함하라. (5) 설명만 출력하라.")
    fb = ("\n이전 답변의 검증 실패 사유: " + "; ".join(feedback) + "\n위 문제를 고쳐 다시 작성하라.") if feedback else ""
    return f"{rules}{fb}\nJSON:\n{json.dumps(rec, ensure_ascii=False)}"


def ollama_generate(prompt: str) -> str:
    """로컬 Ollama 호출 (temperature 0, seed 고정). 실패 시 예외를 그대로 올린다."""
    host = os.environ.get("OLLAMA_HOST", C.OLLAMA_HOST_DEFAULT).rstrip("/")
    body = {"model": os.environ["OLLAMA_MODEL"], "prompt": prompt, "stream": False,
            "options": {"temperature": 0, "seed": C.RANDOM_SEED}}
    req = urllib.request.Request(host + "/api/generate", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8")).get("response", "")


def explain(rec: dict, generate=None) -> dict:
    """한 evidence 레코드 설명. generate: 프롬프트→텍스트 함수(없으면 OLLAMA_MODEL 설정 시 Ollama)."""
    if generate is None and os.environ.get("OLLAMA_MODEL"):
        generate = ollama_generate
    attempts, llm_error, feedback = [], None, None
    if generate is not None:
        for _ in range(1 + C.LLM_MAX_RETRIES):
            try:
                text = _parse_llm(generate(_prompt(rec, feedback)))
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, json.JSONDecodeError) as e:
                llm_error = f"{type(e).__name__}: {e}"
                print(f"WARNING: LLM 호출 실패 — 템플릿 설명으로 대체: {llm_error}")
                break
            v = validate_explanation(text, rec)
            attempts.append({"generator": "llm", "text": text, "FINAL": v["FINAL"], "violations": v["violations"]})
            if v["FINAL"] == "PASS":
                return {"generator": "llm", "final_text": text, "attempts": attempts, "llm_error": None}
            feedback = v["violations"]
    text = template_explain(rec)
    v = validate_explanation(text, rec)
    attempts.append({"generator": "template", "text": text, "FINAL": v["FINAL"], "violations": v["violations"]})
    gen = "template" if generate is None else ("template(llm_error)" if llm_error else "template(llm_rejected)")
    return {"generator": gen, "final_text": text, "attempts": attempts, "llm_error": llm_error}


def generate_explanations(recs: list[dict] | None = None, generate=None) -> list[dict]:
    recs = recs or json.loads((C.JSON_DIR / "dong_results.json").read_text(encoding="utf-8"))
    items = [("dong", r["region_id"], r) for r in recs] + [("policy", r["policy"], r) for r in build_policy_evidence()]
    out = []
    for kind, rid, rec in items:
        e = explain(rec, generate)
        out.append({"kind": kind, "id": rid, "evidence": rec, **e})
    (C.JSON_DIR / "explanations.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    gens = pd.Series([o["generator"] for o in out]).value_counts().to_dict()
    print(f"explanations: {len(out)} ({gens})")
    return out
