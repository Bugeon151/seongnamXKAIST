# 방법론 (methodology) — 구현과 1:1 대응

모든 파라미터는 `src/config.py` 한 곳에서 관리한다. 결과 수치는 `outputs/presentation/12_key_numbers.json`에서 확인한다.

## 0. 데이터·전처리 (`src/data_validation.py`, `src/preprocessing.py`)
- 원본(`data/raw`)은 읽기 전용. 검증 → 전처리 → 가공 데이터 검증 순서로 실행되며, FAIL이면 파이프라인이 중단된다.
- 인구: 시·구 집계행(코드 끝 5자리 00000) 제거, 기준월 = 데이터 내 최신월(2026년 08월) 검증.
- 버스정류장: 완전 공백행 제거, 좌표 수치화, 행정동 폴리곤 point-in-polygon 결합. 경계 밖 2건은 삭제하지 않고
  최근접 동·거리를 `outputs/boundary_exceptions.csv`에 기록.
- 의료기관: 공백 정규화, (기관명+주소) 기준 고유시설 ID(MED_xxxxx). 지오코딩은 `src/geocoding.py`(Nominatim).
- 재가장기요양기관: 급여종류 플래그, 시설이용형(주야간보호·단기보호) 여부, 정원. 지오코딩 동일 방식.
- 지오코딩 결과(외부 API 실측)는 버전관리되는 입력으로 취급하고, 매 실행 시 ID·기관명·주소 1:1 일치를 검사한다.
- 원본 무수정 검증: `data/SHA256SUMS`(원본 전체 + 지오코딩 입력 3개)의 기준 해시와 실제 파일을 매 실행 대조한다.
  해시는 줄바꿈(CRLF→LF) 정규화 후 SHA-256이라 OS·git 설정과 무관하게 같은 값이 나온다. 불일치 시 1단계 FAIL.

## 1. 접근성 (`src/accessibility.py`, `src/network.py`)
- 수요점 i: 행정동 수요 대표점(50) = 동 내 버스정류장 medoid(같은 동 정류장까지 직선거리 합 최소, 정류장 없으면 폴리곤
  내부 대표점, `src/network.py::demand_points`). 수요량 P_i = 65세 이상 인구. 폴리곤 중심점 기준 결과와의 비교는
  `accessibility_summary.json`의 `demand_point`에 기록한다.
- 거리: OSM 보행망 최단경로(m, primary). 대표점·시설은 최근접 노드에 스냅하고 스냅 직선거리를 더한다.
  비교용으로 직선거리도 계산한다.
- E2SFCA (Luo & Qi 2009):
  `R_j = S_j / Σ_i W(d_ij)·P_i`,  `A_i = Σ_j W(d_ij)·R_j`,
  W: 0–1km 1.0, 1–2km 0.6, 2–3km 0.25, 3km 초과 0. 캐치먼트 내 수요가 없는 시설은 R_j = 0.
  성질: `Σ_i P_i·A_i = Σ_{도달 가능 시설} S_j` (공급 보존, 테스트로 검증).
- 공급량: 의료 S1 = 고유시설 1개당 1(병상·의사수 없음, proxy), S2 = 종별 가중(민감도), 복지 = 시설이용형 정원(명).
  방문형 급여는 제공자가 가정을 방문하므로 거리 기반 접근성에 쓰지 않는다.
- 대중교통: 고령 1,000명당 동 내 버스정류장 수(+ 최근접 정류장 보행거리 참고).
- 비교: 최근접 거리 vs E2SFCA, 직선 vs 네트워크 E2SFCA 순위 상관·순위 이동 동 (`accessibility_method_comparison.csv`).

## 2. MVI (`src/mvi.py`)
```
Need_i   = mean(minmax(65세 이상 인구), minmax(고령인구 비율))
Access_i = mean(minmax(의료 E2SFCA), minmax(복지 E2SFCA), minmax(고령 1천명당 정류장))
MVI_i    = Need_i^(2·w_need) × (1 − Access_i)^(2·w_access)       primary(0.5, 0.5) = Need × (1 − Access)
```
- 곱 형식: 수요와 접근 결핍이 **동시에** 큰 동을 높게 평가(Need-Access mismatch).
- 원인 분해(`mvi_components.csv`): 원자료 → 정규화 구성요소 → MVI, 가장 낮은 접근성 요인.
- Ablation(`mvi_ablation.csv`): 의료·복지·교통 각각 제거, Need 단독, 결핍 단독, 가중합 형식 — 순위 상관·상위 10 중복.
- 이상치(|z|>3)는 제거하지 않고 보고하며, 순위 정규화 시나리오로 영향을 점검한다.

## 3. 공간통계 (`src/spatial_statistics.py`)
- Queen 인접, 행표준화. 검증: 섬 없음, 연결요소 1, 대칭.
- Global Moran's I + Local Moran's I(LISA), 999회 조건부 순열, seed 42 고정, p < 0.05만 HH/HL/LH/LL.

## 4. 강건성·Robust Hotspot (`src/sensitivity.py`, `src/hotspot.py`)
- 격자 36 = MVI 가중치 3 × 거리감쇠(밴드 ×0.75/1.0/1.25) 3 × 결합형식(곱/가중합) 2 × 정규화(min-max/순위) 2.
  시나리오마다 접근성→MVI→Local Moran을 다시 계산.
- 동별 HH·유의 출현 비율 ≥ 0.8 → **안정 Robust Hotspot**, 0.2~0.8 → 불안정(조건부).
- 순위 Spearman(primary 대비·쌍별), HH 집합 Jaccard, 상위 10 출현 비율, 'primary만 봤을 때 hotspot이지만 안정적이지 않은 동'.
- DBSCAN(eps 2km)은 안정 hotspot을 운영 권역으로 묶는 보조 도구(유의성 판단 아님).

## 5. MCLP·baseline (`src/optimization.py`, `src/baseline.py`, `src/kpi.py`)
- 후보: 행정동 매칭 검증된 실존 버스정류장. 커버: 정류장–동 수요 대표점 직선거리 ≤ 1.5km. 수요: 고령인구×MVI.
- `max Σ vuln_w·y  s.t. Σx = k, y_i ≤ Σ_{j∈N(i)} x_j` — CBC, Optimal일 때만 최적해로 표기.
- 동점 처리: 1단계 최적값을 고정하고 '선정 정류장과 커버 수요 간 취약가중 평균거리' 합을 최소화(결정적).
- 커버리지 곡선 k=1..10, baseline(무작위 20 seeds, 취약수요 상위 동 우선, 탐욕) — 동일 KPI 함수.

## 6. 정책 A/B/C·노선 (`src/policy_scenarios.py`, `src/routing.py`)
| | 목적함수 | 제약 | 결정변수 |
|---|---|---|---|
| A 공공효과 | max Σ vuln_w·y | Σx = 5 | x, y |
| B 형평성 | max t (+1e-4·전체커버) | Σx = 5, t ≤ 구별 취약커버율 | x, y, t |
| C 운영효율 | min Σ_구 순환노선 거리(vehicle-km) | 취약커버율 ≥ 정책 A, 1 ≤ Σx ≤ 10 | x와 구별 순환노선(방문 순서) |
- C 대표안 = **정책 A와 같은 공공효과를 최소 운행거리로** (같은 산출·최소 투입). 별도로 예산 L = 10~80km마다
  '노선거리 ≤ L에서 커버 최대'(커버리지 Orienteering)를 풀어 (vehicle-km, 커버리지) Pareto 곡선과 비지배 여부를 보고한다.
- 풀이: 구별 후보 8개로 축약(① MCLP k=1..10·Maximin 최적해 등장 정류장을 빈도순 우선,
  ② 동일 커버집합 대표·지배 제거·커버 수요 순으로 채움) → 구별 256개 부분집합의 최단 순환거리(Held-Karp)
  → 3개 구 조합 1,677만 개 **완전탐색** → 축약 후보 내 정확해(시간제한 없음, 결정적).
- 구별 8개 상한 때문에 축약 후보가 정책 A의 해를 항상 포함하지는 않는다(현재 데이터에서도 미포함 —
  `policy_results.json`의 `A_solution_in_reduced_candidates`). 그래서 A의 해(목표 커버를 자명하게 만족)를 같은
  Held-Karp 거리로 평가해 비교 후보에 넣는다 → 대표안의 vehicle-km ≤ A가 구조적으로 보장된다. 전체 후보 전역최적은 아니다.
- 노선(모든 정책 공통): 구 보건소 거점, OSM 차량망(일방통행 반영) 최단거리, Held-Karp 정확 TSP + 차량 분할 DP로
  총 vehicle-km 최소, 순환시간 = 거리/20km/h + 정차 3분×정류장 ≤ 90분(가정, What-if에서 변경 가능).
- KPI: 취약수요·고령인구 커버, 상위 10 취약동 커버, hotspot 커버, 최소 구 커버·구간 격차, vehicle-km, 최대 순환시간,
  차량 수, 10km당 커버, 서비스 정류장까지 보행거리.
- Before/After: before = 전담 서비스 없음. 동·구별 서비스 수혜 고령인구, 미커버 취약수요, 서비스 정류장 보행거리의
  baseline/after/차이/증감률.

## 7. 근거제약형 설명·grounding (`src/explanation.py`, `src/grounding_validator.py`)
- 분석 결과 → evidence JSON(JSON Schema 검증) → 템플릿 또는 LLM(Ollama, 선택) → 파싱 → 검증 → 채택/재생성/템플릿 대체.
- 검증: 숫자(KPI 라벨·단위 결합, 반올림·백분율 허용), 지역명(원자료 어휘 + 분석 범위 밖 인접 지역), 행정코드, 정책명,
  입력에 없는 지표, 인과 표현, 금지 단정·의사결정 권고 표현. 최종 채택 설명은 모두 PASS여야 파이프라인이 통과한다.
- 우회 방지: 지역명 속 숫자만 가리고 한글·단위(km)·문장끝 마침표에 붙은 숫자도 추출, `%`·`km`·`분`·`대`가 붙은 숫자는
  해당 KPI 라벨에 결합되어야 하며(우연한 반올림 일치로 통과 불가), 정책 설명의 비교 수치는 문장에 언급된 정책의 값으로만
  대조한다(다른 정책 값 끼워넣기 탐지). 적대적 사례는 `tests/test_grounding.py`에 고정.

## 8. 재현성
- seed: Moran 42(전역 Moran은 numpy 전역 난수 고정), 무작위 baseline 42. 동점 해소로 최적화 해 결정적.
- `run_metadata.json`: git SHA, 패키지 버전, 파라미터, 원천 파일 SHA-256·기준일, 단계별 소요시간.
  `git_sha`는 산출물을 만든 **코드 커밋**이다. 릴리스는 '검증된 코드 커밋에서 실행 → 산출물·보고서만 담은 보고 커밋'
  순서이므로 최종 HEAD가 보고 커밋이면 `git_sha`가 그 부모 커밋을 가리키는 것이 정상이다
  (`code_modified_since_commit`은 src·스크립트·대시보드·requirements·원본 데이터 변경 여부).
