"""Grounded Explanation Validator — 설명 문장을 입력 JSON(근거)과 자동 대조한다.

검사 항목 (하나라도 위반하면 FINAL=FAIL, violations에 사유 기록):
 1. 숫자      텍스트의 모든 숫자는 입력 JSON 값(또는 선언된 상수)이어야 한다.
              KPI 라벨·단위가 붙은 숫자는 '그 KPI의 값'과 일치해야 한다(값 뒤바꿈 탐지).
              입력값을 같은 소수자리로 반올림한 표기와 비율(0~1)의 백분율 표기(×100)는 허용.
 2. 지역명    행정동·법정동·구 이름 어휘(원자료 기반)에서, 입력 레코드에 없는 지역이 등장하면 FAIL.
              존재하지 않는 'OO숫자동' 형태도 FAIL. 10자리 행정코드는 입력 region_id와 같아야 한다.
 3. 정책명    '정책 X'/'Policy X'/정책 이름은 입력에 있는 정책만 허용.
 4. KPI       입력에 없는 지표(예: Gi*, 소득)를 언급하면 FAIL.
 5. 주장      금지 단정 표현, 의사결정 권고 표현(채택해야 등 — 최종 결정은 사람),
              입력에 없는 인과 주장(때문·로 인해·탓·기인 등)은 FAIL.
숫자 추출: 지역명(신흥1동 등)을 가린 뒤 한글·단위(km 등)에 붙은 숫자도 추출한다.
단위 규칙: %·km·분·대(차량)가 붙은 숫자는 대응 KPI 라벨에 결합되어야 하며, 결합되지 않으면
           '아무 입력값과 우연히 같은 반올림'으로 통과시키지 않고 FAIL로 처리한다.
한계: 문장의 의미적 뉘앙스 전체를 검증하지는 않는다(숫자·개체·지표·표현 규칙 기반).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import pandas as pd

from src import config as C

# 문자열 속 숫자(입력 JSON 인용 허용용): 영문 식별자 일부(E2SFCA·gemma3)와 한글 붙은 숫자는 제외
NUM_RE = re.compile(r"(?<![A-Za-z가-힣0-9_.])-?\d+(?:,\d{3})*(?:\.\d+)?(?![0-9_.]|[A-Za-z])")
# 설명문 속 숫자: 지역명을 가린 뒤 추출하므로 한글 뒤 숫자(예: '인구7458명')도 잡는다.
# 뒤에 오는 영문은 단위(km, m)만 허용하고 문장 끝 마침표('0.403.')도 숫자 경계로 본다.
TEXT_NUM_RE = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:,\d{3})*(?:\.\d+)?(?![0-9_]|\.\d|(?!km|m(?![A-Za-z]))[A-Za-z])")
CODE_RE = re.compile(r"(?<!\d)\d{10}(?!\d)")
NUMBERED_DONG_RE = re.compile(r"[가-힣]{1,4}\d동")
BANNED_CLAIMS = ["완전히 제거", "100% 보장", "반드시 성공", "인과관계가 입증", "통계적으로 인과",
                 "hallucination이 없다", "완벽하게 예측", "확실히 해결", "100% 해소", "완전히 해소"]
# 의사결정 권고 — 시스템은 대안과 trade-off만 제시하고 선택은 사람이 한다
DECISION_CLAIMS = ["반드시", "채택해야", "선택해야", "도입해야", "시행해야", "가장 좋은 정책", "최선의 정책",
                   "최적의 정책", "우월한 정책"]
CAUSAL_PATTERNS = ["때문", "로 인해", "으로 인해", "원인은", "원인이다", "원인으로", "초래", "야기", "탓",
                   "기인", "덕분", "인과적", "영향으로", "의 결과이다", "의 결과로"]
UNSUPPORTED_KPI_TERMS = ["Gi*", "Getis", "소득", "사망률", "만족도", "교통사고", "범죄"]
# 분석 범위(성남시) 밖 인접 지역 — 원자료 어휘에 없으므로 별도로 막는다
OUT_OF_SCOPE_PLACES = ["서울", "송파", "강남구", "서초", "용인", "수지구", "하남", "경기 광주", "광주시", "의왕",
                       "과천", "수원"]
# 이 단위가 붙은 숫자는 대응 KPI 라벨에 결합되어야 한다 (결합 실패 시 FAIL)
STRICT_UNITS = [("%", r"^\s*%"), ("km", r"^\s*km(?!\s*당)"), ("분", r"^\s*분(?![석당배포류야])"),
                ("대", r"^\s*대(?![상비중규형학로표부체신략])")]
PER_KM_RE = re.compile(r"^\s*km\s*당")
PER_KM_UNITS = {10.0}  # KPI vuln_coverage_per_10km의 분모 — '10km당'만 허용
TOL = 1e-6
# 템플릿에 쓰이는 선언된 상수 (JSON 지표가 아니나 허용 — 출처를 함께 보고한다)
KNOWN_CONSTANTS = {65.0: "age_65_threshold", float(C.N_DONG): "n_dongs",
                   float(C.TOP_K_VULNERABLE): "top_k_vulnerable", float(C.K_MAIN): "k_main",
                   float(C.MORAN_ALPHA): "moran_alpha", float(C.MORAN_PERMUTATIONS): "moran_permutations"}
POLICY_NAMES = {"A": ["공공효과"], "B": ["형평성", "형평"], "C": ["운영효율"]}

# (라벨 핵심 정규식, 라벨~숫자 사이 허용 글자 수, 숫자 뒤 단위 정규식 또는 None, 대상 필드 목록)
KPI_BINDINGS = [
    (r"(65세 이상|고령)\s*인구", 6, r"^\s*명", ["elderly_population", "covered_elderly"]),
    (r"(비율|비중|고령화율)", 6, r"^\s*%", ["elderly_ratio"]),
    (r"(시나리오|격자)\s*(중|에서)?", 4, r"^\s*%", ["hotspot_share"]),
    (r"취약\s*(고령)?\s*수요\S*", 3, r"^\s*%", ["vuln_coverage"]),
    (r"(커버|커버리지)", 8, r"^\s*%", ["vuln_coverage", "elderly_coverage", "min_gu_coverage"]),
    (r"차량", 8, r"^\s*대", ["vehicles_used"]),
    (r"(거점|정류장)", 6, r"^\s*개", ["n_new_stops"]),
    (r"(순환|소요)\s*(시간)?", 8, r"^\s*분", ["max_cycle_min"]),
    (r"", 0, r"^\s*km(?!\s*당)", ["total_vehicle_km"]),
    (r"km\s*당\s*취약\s*(고령)?\s*수요\s*커버율?", 4, None, ["vuln_coverage_per_10km"]),
    (r"(구별 최소|최소 구)\s*커버율?", 4, None, ["min_gu_coverage"]),
    (r"취약\s*(고령)?\s*수요\s*커버율?", 12, None, ["vuln_coverage"]),
    (r"MVI", 20, None, ["mvi"]),
    (r"", 0, r"^\s*위", ["mvi_rank", "need_rank", "deficit_rank"]),
    (r"(p\s*=|p값|유의확률)", 4, None, ["local_moran_p"]),
    (r"(Need|수요 지수)", 8, None, ["need"]),
    (r"(Access|종합 접근성)", 8, None, ["access"]),
    (r"의료", 12, None, ["access_medical", "med_e2sfca_network"]),
    (r"복지", 16, None, ["access_welfare", "wel_e2sfca_network"]),
    (r"(대중교통|교통)", 12, None, ["access_transit", "transit_stops_per_1000_elderly"]),
    (r"(운행|노선)", 8, r"^\s*km", ["total_vehicle_km"]),
    (r"접근성", 8, None, ["access"]),  # 구성요소 라벨 없이 '접근성(지수)'만 쓴 경우 = 종합 Access (가장 낮은 우선순위)
]


ENUM_MARKERS = ["각각", "모두", "순서대로", "차례로"]


def _sentence_left(text: str, pos: int) -> str:
    """pos 앞쪽의 같은 문장 부분 (숫자 소수점이 아닌 마침표·물음표·느낌표를 문장 경계로 본다)."""
    cut = 0
    for m in re.finditer(r"(?<!\d)[.?!](?=\s|$)", text[:pos]):
        cut = m.end()
    return text[cut:pos]


def _bind(left: str, right: str, flat: dict, sentence_left: str = "") -> list[str] | None:
    """숫자에 대응하는 KPI 필드.

    - 나열문('각각'·'모두'·'순서대로'·'차례로')은 같은 문장 속 모든 라벨의 필드 합집합으로 결합
    - 그 외에는 단위가 붙은 규칙(명·%·위·km)을 우선 적용한 뒤, 라벨 바로 뒤 숫자 규칙 적용
    """
    ctx = sentence_left or left
    if any(m in ctx for m in ENUM_MARKERS):
        fields = [f for core, _, unit, fs in KPI_BINDINGS if core and re.search(core, ctx)
                  and (unit is None or re.match(unit, right)) for f in fs]
        if fields and _field_values(flat, fields):
            return fields
    ordered = sorted(KPI_BINDINGS, key=lambda b: b[2] is None)  # 단위 규칙 우선 (안정 정렬)
    for core, gap, unit, fields in ordered:
        if re.search(rf"{core}[^0-9]{{0,{gap}}}$", left) and (unit is None or re.match(unit, right)):
            if _field_values(flat, fields):
                return fields
    return None


@lru_cache(maxsize=1)
def place_vocabulary() -> dict:
    """원자료 기반 지역명 어휘: 행정동 50 + 법정동 + 구."""
    area = pd.read_csv(C.RAW_AREA, encoding="utf-8-sig", dtype=str)
    legal = sorted({x.strip() for v in area["비고(관할 법정동)"].dropna() for x in v.split("+")})
    admin = sorted(area["동 명"].str.strip())
    gus = sorted(area["구청명"].str.strip().unique())
    return {"admin": admin, "legal": legal, "gu": gus, "all": sorted(set(admin) | set(legal) | set(gus))}


def _mask_places(text: str, vocab: list[str]) -> str:
    """지역명 속 숫자(신흥1동·상대원9동 등)를 같은 길이의 기호로 가린다 — 숫자 검사 대상에서만 제외(위치 보존).

    가짜 'OO숫자동'은 여기서 가려지지만 지역명 검사(NUMBERED_DONG_RE)에서 별도로 FAIL 처리된다.
    """
    def blank(m: re.Match) -> str:
        return "□" * len(m.group(0))
    for p in sorted((p for p in vocab if any(ch.isdigit() for ch in p)), key=len, reverse=True):
        text = re.sub(re.escape(p), blank, text)
    return NUMBERED_DONG_RE.sub(blank, text)


def _flatten_numbers(obj, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten_numbers(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten_numbers(v, f"{prefix}{i}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix.rstrip(".")] = float(obj)
    return out


def _string_numbers(obj) -> list[float]:
    """입력 JSON의 문자열 값 속 숫자(예: method_version 'mobilens-2.0'의 2.0) — 인용은 허용."""
    out: list[float] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _string_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _string_numbers(v)
    elif isinstance(obj, str):
        out += [float(t.replace(",", "")) for t in NUM_RE.findall(obj)]
    return out


def _decimals(tok: str) -> int:
    return len(tok.split(".")[1]) if "." in tok else 0


def _matches(tok: str, value: float, pct: bool) -> bool:
    raw = tok.replace(",", "")
    n, d = float(raw), _decimals(raw)
    cands = [value] + ([value * 100] if pct and 0 <= value <= 1 else [])
    return any(abs(n - v) <= max(TOL, abs(v) * 1e-9) or abs(n - round(v, d)) < 1e-9 for v in cands)


def _scoped_values(flat: dict[str, float], fields: list[str], rec: dict, sentence_left: str) -> list[float]:
    """정책 evidence는 compare에 A/B/C 값이 함께 있으므로, 결합된 KPI 값을 문장 속 정책에 한정한다.

    같은 문장에서 가장 최근에 언급된 '정책 X'의 값(compare.X, 자기 정책이면 kpis도)만 허용하고,
    정책 언급이 없으면 이 레코드 자신의 값(kpis)만 허용한다 → 다른 정책 값 끼워넣기 탐지.
    """
    vals = [(k, v) for k, v in flat.items() if k.split(".")[-1] in fields]
    if rec.get("kind") == "policy":
        own = rec.get("policy")
        named = re.findall(r"(?:정책|Policy)\s*([A-Z])(?![A-Za-z])", sentence_left)
        target = named[-1] if named else own
        scoped = [v for k, v in vals if k.startswith(f"compare.{target}.")
                  or (target == own and not k.startswith("compare."))]
        if scoped:
            return scoped
    return [v for _, v in vals]


def _field_values(flat: dict[str, float], fields: list[str]) -> list[float]:
    return [v for k, v in flat.items() if k.split(".")[-1] in fields]


def validate_explanation(text: str, rec: dict, known_dongs: list[str] | None = None) -> dict:
    """설명 text를 근거 레코드 rec와 대조.

    rec에서 사용하는 키: 숫자 값 전체, region_id, region_name, gu, legal_dongs, allowed_places,
    policies_available. known_dongs는 하위호환용(무시 — 지역명 어휘는 원자료에서 구성).
    """
    violations: list[str] = []
    flat = _flatten_numbers(rec)
    flat.update({f"_str.{i}": v for i, v in enumerate(_string_numbers(rec))})
    numeric: dict[str, str] = {}
    vocab = place_vocabulary()["all"]
    for m in TEXT_NUM_RE.finditer(_mask_places(text, vocab)):
        tok = m.group(0)
        raw = tok.replace(",", "")
        if CODE_RE.fullmatch(raw) or (raw.isdigit() and 2020 <= int(raw) <= 2030):
            continue  # 행정코드(아래 별도 검사)·연도
        if raw.startswith("-") and m.start() > 0 and text[m.start() - 1].isalnum():
            tok, raw = tok[1:], raw[1:]  # 'mobilens-2.0'처럼 하이픈 뒤 숫자는 음수가 아니다
        left, right = text[max(0, m.start() - 30):m.start()], text[m.end():m.end() + 4]
        pct = right.lstrip().startswith("%")
        if PER_KM_RE.match(right):  # 'NNkm당' — 비율 KPI의 분모 단위(10km당만 허용)
            ok = any(abs(float(raw) - c) <= TOL for c in PER_KM_UNITS)
            numeric[tok] = "PASS(const:per_km_unit)" if ok else "FAIL"
            if not ok:
                violations.append(f"'km당' 단위 숫자 {tok}는 선언된 상수가 아님")
            continue
        fields = _bind(left, right, flat, _sentence_left(text, m.start()))
        bound = (fields, _scoped_values(flat, fields, rec, _sentence_left(text, m.start()))) if fields else None
        if bound:
            ok = any(_matches(tok, v, pct) for v in bound[1])
            numeric[tok] = f"{'PASS' if ok else 'FAIL'}(kpi:{bound[0][0]})"
            if not ok:
                violations.append(f"숫자 {tok}가 KPI '{bound[0][0]}' 값과 불일치")
            continue
        unit = next((u for u, pat in STRICT_UNITS if re.match(pat, right)), None)
        if unit:
            numeric[tok] = f"FAIL(unbound_unit:{unit})"
            violations.append(f"단위 '{unit}'가 붙은 숫자 {tok}가 어떤 KPI인지 라벨로 결합되지 않음")
            continue
        if any(_matches(tok, v, pct) for v in flat.values()):
            numeric[tok] = "PASS(src)"
        elif any(abs(float(raw) - c) <= TOL for c in KNOWN_CONSTANTS):
            numeric[tok] = f"PASS(const:{KNOWN_CONSTANTS[min(KNOWN_CONSTANTS, key=lambda c: abs(float(raw) - c))]})"
        else:
            numeric[tok] = "FAIL"
            violations.append(f"입력에 없는 숫자 {tok}")

    region = rec.get("region_name")
    allowed = set(rec.get("allowed_places", [])) | set(rec.get("legal_dongs", []))
    allowed |= {x for x in (region, rec.get("gu")) if x}
    # 앞은 단어 경계, 뒤는 조사(의·보다·에 등)가 붙을 수 있으므로 숫자만 금지
    mentioned = sorted({p for p in vocab if re.search(rf"(?<![가-힣0-9]){re.escape(p)}(?!\d)", text)})
    mentioned += [p for p in OUT_OF_SCOPE_PLACES if p in text]
    entity = {p: ("PASS" if p in allowed else "FAIL") for p in mentioned}
    violations += [f"입력에 없는 지역명 {p}" for p, v in entity.items() if v == "FAIL"]
    fake = [t for t in NUMBERED_DONG_RE.findall(text) if t not in vocab]
    violations += [f"존재하지 않는 행정동 {t}" for t in fake]
    bad_codes = [c for c in CODE_RE.findall(text) if c != str(rec.get("region_id"))]
    violations += [f"입력과 다른 행정코드 {c}" for c in bad_codes]
    if region and region not in text:
        violations.append(f"대상 지역명 {region} 누락")

    allowed_pol = set(rec.get("policies_available", []))
    pol_mentions = set(re.findall(r"(?:정책|Policy)\s*([A-Z])(?![A-Za-z])", text))
    pol_mentions |= {k for k, names in POLICY_NAMES.items() if any(n in text for n in names)}
    bad_pol = sorted(pol_mentions - allowed_pol)
    violations += [f"입력에 없는 정책 {p}" for p in bad_pol]

    bad_kpi = [t for t in UNSUPPORTED_KPI_TERMS if t in text]
    violations += [f"입력에 없는 지표 {t}" for t in bad_kpi]
    claims = [b for b in BANNED_CLAIMS if b in text]
    decision = [d for d in DECISION_CLAIMS if d in text]
    causal = [c for c in CAUSAL_PATTERNS if c in text]
    violations += ([f"금지 단정 표현 '{c}'" for c in claims] + [f"의사결정 권고 표현 '{d}'" for d in decision]
                   + [f"입력에 없는 인과 주장 '{c}'" for c in causal])

    return {"region_id": rec.get("region_id"), "region_name": region, "numeric": numeric,
            "numeric_pass": all(v.startswith("PASS") for v in numeric.values()),
            "unmatched_numbers": [k for k, v in numeric.items() if v.startswith("FAIL")],
            "mentioned_places": mentioned, "entity": entity, "unmatched_region_codes": bad_codes,
            "entity_pass": not any(v == "FAIL" for v in entity.values()) and not fake and not bad_codes
            and (region is None or region in text),
            "policy_mentions": sorted(pol_mentions), "policy_pass": not bad_pol,
            "unsupported_kpis": bad_kpi, "banned_claims_found": claims + decision, "causal_claims_found": causal,
            "claims_pass": not claims and not decision and not causal, "violations": violations,
            "FINAL": "PASS" if not violations else "FAIL"}


def run_validation() -> dict:
    """explanations.json의 최종 설명 전체를 재검증한 결과를 저장한다 (FAIL이 남으면 예외)."""
    exps = json.loads((C.JSON_DIR / "explanations.json").read_text(encoding="utf-8"))
    results = [{**validate_explanation(e["final_text"], e["evidence"]),
                "kind": e["kind"], "id": e["id"], "generator": e["generator"]} for e in exps]
    summary = {"n": len(results), "pass": sum(r["FINAL"] == "PASS" for r in results),
               "fail": sum(r["FINAL"] == "FAIL" for r in results),
               "by_generator": pd.Series([r["generator"] for r in results]).value_counts().to_dict()}
    payload = {"summary": summary, "results": results}
    (C.JSON_DIR / "grounding_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    print(f"grounding: {summary['pass']}/{summary['n']} PASS {summary['by_generator']}")
    if summary["fail"]:
        raise RuntimeError(f"최종 설명 중 grounding FAIL {summary['fail']}건 — 폴백 로직 확인 필요")
    return payload


def intentional_failure_demo() -> dict:
    """고의 오류 데모: 올바른 설명은 PASS, 숫자·지역·정책·인과·지표를 변조하면 FAIL이어야 한다."""
    exps = json.loads((C.JSON_DIR / "explanations.json").read_text(encoding="utf-8"))
    e = next(x for x in exps if x["kind"] == "dong")
    rec, good = e["evidence"], e["final_text"]
    other = next(n for n in place_vocabulary()["admin"]
                 if n != rec["region_name"] and n not in rec.get("legal_dongs", []))
    m = re.search(r"\d+\.\d+", good)
    cases = {
        "correct": good,
        "tampered_number": good[:m.start()] + str(round(float(m.group(0)) + 9.0, 4)) + good[m.end():],
        "wrong_region": good + f" 인접한 {other}도 같은 수준이다.",
        "unknown_policy": good + " 정책 D를 적용하면 개선된다.",
        "causal_claim": good + " 이는 버스 노선 부족 때문이다.",
        "unsupported_kpi": good + " Gi* 분석에서도 핫스팟이다.",
    }
    res = {k: validate_explanation(t, rec) for k, t in cases.items()}
    demo = {k: {"FINAL": v["FINAL"], "violations": v["violations"]} for k, v in res.items()}
    demo["tampered_text"] = cases["tampered_number"]
    demo["expected"] = "correct=PASS, 나머지=FAIL"
    demo["demo_pass"] = res["correct"]["FINAL"] == "PASS" and all(
        res[k]["FINAL"] == "FAIL" for k in cases if k != "correct")
    (C.JSON_DIR / "grounding_failure_demo.json").write_text(json.dumps(demo, ensure_ascii=False, indent=2),
                                                           encoding="utf-8")
    print(f"failure demo: demo_pass={demo['demo_pass']}")
    if not demo["demo_pass"]:
        raise RuntimeError(f"grounding 고의 오류 데모 실패: {demo}")
    return demo
