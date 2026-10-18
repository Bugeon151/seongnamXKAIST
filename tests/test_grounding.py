"""grounding validator·LLM 설명 흐름 테스트 — 합성 evidence와 mock LLM으로 결정적 검증."""
from __future__ import annotations

import urllib.error

import pytest
from jsonschema import ValidationError, validate as js_validate

from src.explanation import explain, template_explain, DONG_SCHEMA, _parse_llm
from src.grounding_validator import validate_explanation


@pytest.fixture()
def rec() -> dict:
    return {"kind": "dong", "region_id": "4113357000", "region_name": "상대원1동", "gu": "중원구",
            "legal_dongs": ["상대원동"], "elderly_population": 4321, "elderly_ratio": 23.45,
            "need": 0.8123, "access": 0.2211, "access_medical": 0.3012, "access_welfare": 0.1105,
            "access_transit": 0.2516, "weakest_access": "복지(주야간보호) 접근성", "mvi": 0.6327, "mvi_rank": 2,
            "local_moran_class": "High-High", "local_moran_p": 0.012, "hotspot_class": "stable",
            "hotspot_share": 0.9444, "n_robust_scenarios": 36, "covered_by_policy": {"A": True, "B": False, "C": True},
            "policies_available": ["A", "B", "C"], "method_version": "mobilens-2.0"}


def test_schema_and_template_pass(rec):
    js_validate(rec, DONG_SCHEMA)
    r = validate_explanation(template_explain(rec), rec)
    assert r["FINAL"] == "PASS", r["violations"]


def test_schema_rejects_bad_record(rec):
    bad = dict(rec, mvi=1.7)
    with pytest.raises(ValidationError):
        js_validate(bad, DONG_SCHEMA)


@pytest.mark.parametrize("text,expect", [
    ("상대원1동의 MVI는 0.6327이다.", "PASS"),
    ("상대원1동의 MVI는 0.63이다.", "PASS"),                         # 같은 자리 반올림 허용
    ("상대원1동의 고령인구 비율은 23.45%이다.", "PASS"),
    ("상대원1동은 법정동 상대원동에 속하며 중원구에 있다.", "PASS"),
    ("상대원1동의 MVI는 23.45이다.", "FAIL"),                        # KPI 값 뒤바꿈
    ("상대원1동의 65세 이상 인구는 5000명이다.", "FAIL"),              # 입력에 없는 숫자
    ("상대원1동의 MVI는 0.6327로 금광1동보다 높다.", "FAIL"),           # 입력에 없는 지역
    ("상대원1동과 상대원9동을 비교하면 MVI는 0.6327이다.", "FAIL"),      # 존재하지 않는 동
    ("상대원1동(코드 9999999999)의 MVI는 0.6327이다.", "FAIL"),        # 다른 행정코드
    ("상대원1동은 정책 D 대상이다.", "FAIL"),                           # 입력에 없는 정책
    ("상대원1동은 복지시설 부족 때문에 취약하다.", "FAIL"),              # 인과 주장
    ("상대원1동은 Gi* 핫스팟이다.", "FAIL"),                            # 입력에 없는 지표
    ("MVI는 0.6327이다.", "FAIL"),                                      # 대상 지역명 누락
    ("상대원1동 분석은 mobilens-2.0 방법으로 수행되었다.", "PASS"),      # 입력 문자열 속 숫자(버전)
    ("상대원1동의 복지 접근성은 0.2211이다.", "FAIL"),                   # 실제 LLM 오류 유형: 종합 Access 값을 복지로 표기
    ("상대원1동의 의료 및 복지 접근성은 각각 0.3012와 0.1105이다.", "PASS"),  # '각각' 나열문
    ("상대원1동의 의료 및 복지 접근성은 각각 0.3012와 0.2516이다.", "FAIL"),  # 나열문 속 다른 지표 값(교통)
    ("상대원1동의 의료, 복지, 대중교통 접근성은 순서대로 0.3012, 0.1105, 0.2516이다.", "PASS"),
    ("상대원1동의 종합 접근성 지수와 의료, 복지, 대중교통 접근성 지수는 각각 0.2211, 0.3012, 0.1105, 0.2516이다.", "PASS"),
    ("상대원1동의 MVI는 0.6327이며 지역 내 순위는 2위이다.", "PASS"),                      # 단위(위) 우선
    ("상대원1동의 MVI는 0.6327이며 지역 내 순위는 3위이다.", "FAIL"),
    ("상대원1동의 문제는 완전히 제거된다.", "FAIL"),                     # 금지 단정
])
def test_validator_cases(rec, text, expect):
    r = validate_explanation(text, rec)
    assert r["FINAL"] == expect, (text, r["violations"])


def test_parse_llm_strips_reasoning():
    assert _parse_llm("<think>계산 1234</think>\n**상대원1동**의 MVI는 0.6327이다.") == "상대원1동의 MVI는 0.6327이다."


def test_llm_accepted(rec):
    out = explain(rec, generate=lambda p: "상대원1동(중원구)의 MVI는 0.6327로 50개 동 중 2위이다.")
    assert out["generator"] == "llm" and out["attempts"][0]["FINAL"] == "PASS"


def test_llm_retry_with_feedback(rec):
    calls = []

    def gen(prompt):
        calls.append(prompt)
        return "상대원1동의 MVI는 0.9999이다." if len(calls) == 1 else "상대원1동의 MVI는 0.6327이다."
    out = explain(rec, generate=gen)
    assert out["generator"] == "llm" and len(calls) == 2
    assert "검증 실패 사유" in calls[1] and "0.9999" in calls[1]


def test_llm_rejected_falls_back_to_template(rec):
    out = explain(rec, generate=lambda p: "상대원1동은 교통 부족 때문에 MVI가 0.99이다.")
    assert out["generator"] == "template(llm_rejected)"
    assert out["final_text"] == template_explain(rec)
    assert [a["FINAL"] for a in out["attempts"]] == ["FAIL", "FAIL", "PASS"]


def test_llm_error_is_recorded_not_hidden(rec):
    def boom(prompt):
        raise urllib.error.URLError("connection refused")
    out = explain(rec, generate=boom)
    assert out["generator"] == "template(llm_error)" and "connection refused" in out["llm_error"]


# ---------- 적대적(adversarial) 사례 — 우회 시도는 FAIL, 정상 표현은 PASS ----------
@pytest.mark.parametrize("text,expect", [
    ("상대원1동의 노인 중 81%가 병원에 가기 어렵다.", "FAIL"),           # 라벨 없는 백분율(Need 0.8123≈81%와 우연 일치)
    ("상대원1동의 사업으로 취약성은 100% 해소된다.", "FAIL"),
    ("상대원1동은 버스 노선이 부족하기 때문입니다.", "FAIL"),             # 인과 표현 변형
    ("상대원1동은 대중교통 부족 탓에 취약하다.", "FAIL"),
    ("상대원1동의 취약성은 복지시설 부족에 기인한다.", "FAIL"),
    ("상대원1동은 인과적으로 접근성의 영향을 받는다.", "FAIL"),
    ("상대원1동은 서울 송파구보다 취약하다.", "FAIL"),                   # 분석 범위 밖 지역
    ("상대원1동은 신규 거점 설치가 반드시 필요하다.", "FAIL"),            # 의사결정 권고
    ("상대원1동의 MVI는0.9999이다.", "FAIL"),                            # 한글에 붙은 숫자
    ("상대원1동의 MVI는 0.9999.", "FAIL"),                               # 문장 끝 마침표에 붙은 숫자
    ("상대원1동에서 병원까지 2.5km이다.", "FAIL"),                       # 단위(km) 붙은 근거 없는 거리
    ("상대원1동은 강건성 격자 36개 시나리오 중 94.4%에서 고-고 군집이다.", "PASS"),
    ("상대원1동은 강건성 격자 36개 시나리오 중 70%에서 고-고 군집이다.", "FAIL"),
    ("상대원1동의 MVI는 약 63%이다.", "PASS"),                           # 라벨 결합된 백분율 반올림
])
def test_validator_adversarial_dong(rec, text, expect):
    r = validate_explanation(text, rec)
    assert r["FINAL"] == expect, (text, r["violations"])


@pytest.fixture()
def prec() -> dict:
    k = {"n_new_stops": 5, "vuln_coverage": 0.8653, "elderly_coverage": 0.7731, "covered_elderly": 136653,
         "min_gu_coverage": 0.7745, "total_vehicle_km": 21.47, "max_cycle_min": 60.3, "vehicles_used": 3,
         "vuln_coverage_per_10km": 0.403}
    cmp_ = {"A": {"vuln_coverage": 0.8653, "min_gu_coverage": 0.7745, "total_vehicle_km": 21.47, "vuln_coverage_per_10km": 0.403},
            "B": {"vuln_coverage": 0.8576, "min_gu_coverage": 0.8271, "total_vehicle_km": 20.13, "vuln_coverage_per_10km": 0.426},
            "C": {"vuln_coverage": 0.8714, "min_gu_coverage": 0.7534, "total_vehicle_km": 17.3, "vuln_coverage_per_10km": 0.5037}}
    return {"kind": "policy", "policy": "A", "name": "A_공공효과우선", "goal": "제한된 거점으로 취약 고령수요 커버 최대",
            "kpis": k, "compare": cmp_, "allowed_places": ["분당구", "수정구", "중원구"], "policies_available": ["A", "B", "C"]}


def test_policy_template_passes(prec):
    r = validate_explanation(template_explain(prec), prec)
    assert r["FINAL"] == "PASS", r["violations"]


@pytest.mark.parametrize("old,new", [
    ("21.47km", "12.5km"),               # 단위가 붙은 운행거리 변조
    ("차량 3대", "차량 5대"),             # 다른 KPI(거점 수 5)와 같은 숫자로 바꿔치기
    ("60.3분", "45.0분"),
    ("최소 커버율은 0.7745", "최소 커버율은 0.8653"),   # 다른 KPI 값
    ("커버율은 0.403", "커버율은 0.5037"),             # 다른 정책(C)의 값 끼워넣기
    ("10km당", "5km당"),
])
def test_validator_policy_tampering_detected(prec, old, new):
    good = template_explain(prec)
    assert old in good
    r = validate_explanation(good.replace(old, new), prec)
    assert r["FINAL"] == "FAIL", r


@pytest.mark.parametrize("text,expect", [
    ("정책 A는 차량 3대로 총 21.47km를 운행한다.", "PASS"),
    ("정책 A는 정책 B보다 운행거리가 3km 짧다.", "FAIL"),             # 근거 없는 차이 계산
    ("정책 A는 취약수요의 95%를 커버한다.", "FAIL"),
    ("정책 A는 취약수요의 87%를 커버한다.", "PASS"),
    ("정책 A가 가장 좋은 정책이므로 채택해야 한다.", "FAIL"),          # 결정은 사람
    ("취약수요 커버율은 정책 B 0.8576이다.", "PASS"),                  # 비교 정책 값은 그 정책 이름과 함께
    ("취약수요 커버율은 정책 B 0.8714이다.", "FAIL"),                  # B에 C 값을 붙임
])
def test_validator_policy_claims(prec, text, expect):
    r = validate_explanation(text, prec)
    assert r["FINAL"] == expect, (text, r["violations"])
