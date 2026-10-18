# AI 활용 내역 (ai_usage)

## 시스템 안의 AI / 알고리즘
| 구성요소 | 방법 | 역할 | 결정권 |
|---|---|---|---|
| 접근성 | E2SFCA (OSM 네트워크 거리) | 공급/수요 경쟁을 반영한 잠재 접근성 | 계산 |
| 취약지수 | MVI = Need × (1 − Access) | 수요-접근 불일치 정량화 | 계산 |
| 공간통계 | Local Moran's I, 36 시나리오 강건성, DBSCAN | 유의한 취약 군집·안정성 | 계산 |
| 최적화 | MCLP·Maximin (MILP, CBC), 커버리지 Orienteering(축약 후보 완전탐색), Held-Karp | 입지·노선 대안 생성 | 대안 제시 |
| 설명 | 템플릿(기본) / 로컬 LLM(Ollama, 선택) | 구조화 결과를 문장으로 | 설명만 |
| 검증 | grounding validator (규칙 기반) | 설명의 숫자·지역·정책·지표·인과 표현 대조 | 채택/거부 |

- LLM은 계산하지 않는다. 입력은 JSON Schema로 검증된 evidence JSON뿐이며, 출력은 검증을 통과해야 채택된다.
  실패 시 위반 사유를 알려주고 1회 재생성, 그래도 실패하면 결정적 템플릿으로 대체한다(모든 시도 기록).
- 실제 로컬 모델 측정 기록: `python scripts/llm_grounding_check.py --model <모델>` → `outputs/json/llm_grounding_check.json`.
- 정책의 우열·최종 선택은 사람이 한다. 시스템은 동일 KPI로 대안 간 trade-off를 제시한다.

## 개발 과정의 AI 도구 사용
- 코드 구현·디버깅·테스트 작성·문서 초안에 AI 코딩 도구를 사용했다. 모든 수치는 코드 실행 산출물이며,
  테스트(`pytest`)와 `scripts/preflight.py`로 검증한 결과만 보고한다.
