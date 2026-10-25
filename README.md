# MobiLens Seongnam

**성남시 공개 집계자료로 교통약자의 의료·복지 접근 취약지역을 찾고, 이동지원 거점·노선 대안을 같은 KPI로 비교해
사람의 정책 결정을 돕는 재현 가능한 분석 시스템.**

> AI는 정책을 결정하지 않는다. 검증 가능한 근거와 대안 간 trade-off를 제시한다.
> 2026 제2회 성남 x KAIST AI 경진대회 본선 제출 코드 원본.
> **결과: 학생부 장려상(7위)** — 196개 팀 중 본선 16팀(학생부 10팀), 2026-10-24 발표 (성남시 공고 제2026-2232호).

---

## 1. 문제 정의
- 성남시 65세 이상 인구 **177,033명(19.67%, 2026년 08월)**. 의료·복지 시설과 대중교통은 동마다 고르지 않다.
- "가까운 병원이 있는가"만 보면 **시설을 함께 쓰는 수요(혼잡)**를 놓친다. 실제로 최근접 거리 순위와
  E2SFCA 접근성 순위의 상관은 **ρ = 0.23**에 불과하다(`outputs/json/accessibility_summary.json`).
- 한정된 이동지원 자원을 **어디에, 어떤 노선으로** 둘지는 공공효과·형평·운영효율 사이의 선택이다.

## 2. 전체 구조
```
data/raw (원본, 읽기 전용)          ── 인구 · 버스정류장 · 의료기관 · 재가요양기관 · 행정구역 · 행정동 경계 · OSM 도로망
   │ [1] 데이터 검증 (인코딩·스키마·기준일·좌표·CRS·결측·중복·코드 매칭)            src/data_validation.py
   │ [2] 전처리 → data/processed (+ 지오코딩 결과 연결 검증)                       src/preprocessing.py
   │ [3] 접근성: 의료·복지 E2SFCA(OSM 보행망) + 대중교통, 최근접거리 비교            src/accessibility.py, network.py
   │ [4] MVI = Need × (1 − Access), 원인 분해·ablation                              src/mvi.py
   │ [5] Local Moran's I + 36 시나리오 강건성 → 안정/불안정 Hotspot, DBSCAN 권역       src/spatial_statistics.py, sensitivity.py, hotspot.py
   │ [6] MCLP·baseline → 정책 A(공공효과)/B(형평)/C(운영효율) + 차량노선(OSM 차량망)   src/optimization.py, baseline.py, policy_scenarios.py, routing.py
   │ [7] 근거 JSON → 설명(템플릿 / 선택: 로컬 LLM) → grounding 자동검증              src/explanation.py, grounding_validator.py
   │ [8] 지도·표·발표자료·메타데이터                                                 src/visualization.py, reporting.py
   ▼
outputs/ (결과)  →  대시보드(dashboard/app.py) · CLI 데모(demo.py) · What-if(src/what_if.py)
```

## 3. 데이터 출처
| 데이터 | 제공 | 기준일 |
|---|---|---|
| 고령인구현황 | 성남시 공공데이터 | 2023-09~2026-08 (분석 2026년 08월) |
| 버스정류장 현황 | 성남시 공공데이터 | 2026-04-08 |
| 의료기관 현황 | 성남시 공공데이터 | 2026-03-04 |
| 재가장기요양기관 현황 | 성남시 공공데이터 | 2026-06-05 |
| 행정구역 현황 | 성남시 공공데이터 | 2026-05-15 |
| 행정동 경계 | SGIS 원자료 기반 가공본(vuski/admdongkor) | 2026-07-01 |
| 도로망(보행·차량) | © OpenStreetMap contributors (ODbL) | `data/raw/osm/osm_snapshot_meta.json` |

좌표가 없는 의료기관·재가요양기관은 Nominatim으로 실측 지오코딩했다(성남시 안: 의료 1,237/1,903, 재가요양 215/301).
임의 좌표는 만들지 않았다. 상세: `docs/data_sources.md`, 이용조건: `NOTICE.md`.

## 4. 설치
Python **3.13** 기준(Windows 11에서 검증). 인터넷은 패키지 설치에만 필요하다.
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```
macOS/Linux: `python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt`

- Windows 11 **스마트 앱 컨트롤**이 켜진 PC에서는 일부 최신 휠(scikit-learn 1.9.1·pyogrio 0.13.0·pyarrow 25)의 DLL이
  차단되어 `DLL load failed … 애플리케이션 제어 정책` 오류가 난다. `requirements.txt`는 차단되지 않는 버전으로 고정했고
  산출물 값이 동일함을 확인했다. 새 가상환경 첫 import에서 같은 오류가 한 번 나면 검사 후 통과하므로 다시 실행하면 된다.
- 외부 서비스 없이 재현된다: 지오코딩(Nominatim)·OSM 다운로드·LLM(Ollama)은 기본 실행 경로에서 호출하지 않는다.

## 5. 실행
```powershell
.\.venv\Scripts\python run_pipeline.py              # raw → 전처리 → 분석 → outputs/ 전체 재생성 (약 10초)
.\.venv\Scripts\python run_pipeline.py --validate   # outputs_validation/에 새로 만들고 저장소 outputs/와 값 비교
```
실행 로그는 `[1/8] Data validation … [8/8] Outputs` 단계별 PASS/FAIL과 소요시간을 출력하고
`outputs/logs/pipeline_status.csv`, `outputs/run_metadata.json`(git SHA·패키지 버전·파라미터·seed·원천 파일 SHA-256)에 남긴다.
어느 단계가 실패하면 원인을 출력하고 이후 단계는 실행하지 않는다.

## 6. 데모
```powershell
.\.venv\Scripts\python run_pipeline.py --demo --tamper        # CLI: 문제→취약지역→설명·검증→고의오류 FAIL→정책비교→What-if
.\.venv\Scripts\python run_pipeline.py --what-if policy=B k=7 radius_m=1000   # 파라미터 변경 실제 재계산
.\.venv\Scripts\python -m streamlit run dashboard/app.py      # 대시보드 (개요·MVI·Hotspot·최적화·정책·What-if·설명·데이터품질)
```
대시보드 지도는 OpenStreetMap 배경타일을 쓰므로 오프라인이면 배경만 비고 데이터 레이어는 표시된다.
선택: 로컬 LLM 설명 — Ollama 실행 후 `--llm-model gemma3:4b` (검증 통과한 문장만 채택, 실패 시 템플릿).

## 7. 테스트
```powershell
.\.venv\Scripts\python -m pytest tests -q          # 단위·통합·파이프라인·회귀·grounding·대시보드 (raw부터 전체 실행 포함, 약 20초)
.\.venv\Scripts\python scripts\preflight.py --package   # 제출 전 Gate A~J 자동 점검 → MOBILENS FINAL RELEASE READY
```

## 8. 주요 결과 (모든 수치: `outputs/presentation/12_key_numbers.json`)
| 항목 | 결과 |
|---|---|
| MVI 1위 | 성남동 0.688 (Need 0.80, 대중교통 접근성 최저) — 상위 10개 원인 분해 `02_top_vulnerable_regions.csv` |
| 공간 군집 | Global Moran's I = 0.273 (p = 0.004), Local Moran HH·유의 6개 동 |
| 안정 Robust Hotspot | **4개 동**(금광2동·상대원1동·수진1동·은행2동): 36개 시나리오 중 80% 이상 HH·유의. 불안정 6개 동 구분 |
| 강건성 | 모든 시나리오 순위 상관 ρ ≥ 0.815 |
| MCLP (k=5) | 취약수요 커버 **0.865**, 취약수요 상위 동 우선 배치(0.621) 대비 **+24.4%p**, 무작위 평균 0.345 |

**정책 A/B/C — 동일 KPI**
| 정책 | 목적함수 | 거점 | 취약수요 커버 | 최소 구 커버 | 1회 순환 운행 | 10km당 커버 |
|---|---|---|---|---|---|---|
| A 공공효과 | 취약커버 최대 (k=5) | 5 | **0.865** | 0.774 | 21.5km | 0.403 |
| B 형평성 | 최소 구 커버 최대 (k=5) | 5 | 0.858 | **0.827** | 20.1km | 0.426 |
| C 운영효율 | A 이상 커버를 최소 운행거리로 | 6 | 0.871 | 0.753 | **17.3km** | **0.504** |

세 정책은 서로 다른 정류장을 고른다(정류장 Jaccard A-B 0.25, A-C 0.10). 우열 판단은 의사결정자의 몫이다.
설명 53건 grounding 53/53 PASS, 숫자·지역·정책·인과·지표 변조 5종 모두 FAIL로 탐지. 단위 붙은 숫자 변조·다른 정책 값
끼워넣기·인과 표현 변형·권고 표현 등 적대적 사례는 `tests/test_grounding.py`로 고정 검증한다.

## 9. 주요 산출물
| 경로 | 내용 |
|---|---|
| `outputs/presentation/` | 발표용 12종(문제 요약, 취약 상위 10, 지도 3, 정책 비교, trade-off, before/after, baseline, 강건성, grounding, 핵심 수치) |
| `outputs/tables/mvi_components.csv` | 동별 Need·의료·복지·교통 접근성·MVI 원인 분해 |
| `outputs/tables/accessibility_method_comparison.csv` | 최근접 거리 vs E2SFCA 순위 비교 |
| `outputs/tables/hotspot_frequency.csv`, `robustness_report.json` | 36 시나리오 출현 빈도·Jaccard·순위상관·안정성 분류 |
| `outputs/json/policy_results.json` | 정책별 목적·제약·선정 정류장·노선·KPI·비교 |
| `outputs/tables/policy_C_pareto.csv` | 운행거리 예산별 최대 커버(Pareto, 비지배 검증) |
| `outputs/tables/before_after_by_dong.csv` | 동·정책별 baseline/after/차이/증감률 |
| `outputs/json/explanations.json`, `grounding_validation.json` | 근거 JSON·설명·검증 결과·시도 이력 |
| `outputs/maps/mvi_interactive.html` | 인터랙티브 지도(MVI, 정책 거점·노선) |

## 10. 왜 이 방법인가
- **E2SFCA**: 최근접 거리는 수요 경쟁을 무시한다. 보행 네트워크 거리로 공급/수요비를 계산해 '함께 쓰는 사람 수'까지 반영.
- **MVI 곱 형식**: 수요와 접근 결핍이 *동시에* 큰 곳을 찾기 위한 예선 정의. 가중합 형식은 강건성 시나리오로 비교.
- **Local Moran + 36 시나리오**: 색칠이 아니라 유의성 검정. 가중치·거리감쇠·결합형식·정규화를 바꿔도 남는 곳만 '안정'.
- **MCLP / Maximin / 운행거리 최소화**: 같은 데이터, 다른 목적함수 → 공공효과·형평·효율의 trade-off를 수치로.
- **정확 노선(Held-Karp)**: 거점 수가 작아 근사 대신 정확해. 도로망은 OSM 차량망(일방통행 반영).
- **근거제약형 설명**: LLM은 계산하지 않고 검증된 JSON만 문장화, 규칙 기반 validator가 채택 여부를 결정.

## 11. 개인정보·윤리·AI 사용
- AI의 역할: 공간 AI·최적화(E2SFCA·MVI·LISA·MCLP·노선)는 **계산과 대안 생성**, 설명 계층(템플릿 기본, 로컬 LLM 선택)은
  구조화 근거를 **문장화**만 하고, 규칙 기반 validator가 채택 여부를 결정한다. 상세: `docs/ai_usage.md`.
- 개인 단위 의료·위치·이동 데이터를 사용하지 않는다. 분석 단위는 행정동 집계, 시설은 공개 기관 정보.
- 원자료의 전화번호는 기관 대표번호이며 분석에 쓰지 않는다. API 키·토큰 불필요(`.env`는 선택 설정만, 제출 제외).
- 결과는 '우선순위 후보'이며 인과를 주장하지 않는다. 최종 결정은 사람이 한다.

## 12. 한계
의료 공급은 시설 수 proxy(병상·의사수 없음), 지오코딩 누락(재가요양 시설이용형 정원의 69%만 좌표 확보),
수요점은 동 대표점(동 내 정류장 medoid), 서비스 반경은 직선 1.5km, 노선 속도·정차시간은 가정값,
정책 C는 축약 후보(구별 8개) 내 정확해(전역최적 아님, A의 해와 비교해 운행거리 ≤ A만 보장), **MVI 순위는 Need의 영향이 크다(Need 단독과 ρ = 0.94)**.
전체 목록: `docs/limitations.md`.

## 13. 재현 방법
1. `git clone` 또는 제출 ZIP 압축해제(git 없이도 동작) → 4장 설치 → `python run_pipeline.py --validate`
2. `outputs_validation/COMPARISON.md`에서 저장소 결과와 파일별 비교(값 동일 / 렌더링 차이만 허용)
3. `python -m pytest tests -q` — raw부터 새로 실행한 결과로 불변식을 검사
- 원본 무수정 검증: `data/SHA256SUMS`(줄바꿈 정규화 SHA-256)와 원본·지오코딩 입력을 매 실행 대조한다(불일치 시 1단계 FAIL).
- 모든 난수는 seed 고정, 최적화는 동점 해소 규칙으로 결정적. 지오코딩(`python -m src.geocoding …`)과
  OSM 스냅샷(`scripts/download_osm_network.py`)은 외부 서비스라 재실행 시 달라질 수 있어 결과 파일을 버전관리한다.

최종 제출본의 감사·검증 결과(Gate A~J, ZIP 독립 검증, 핵심 수치 회귀)는 `docs/FINAL_VERIFICATION.md`.

### 릴리스·provenance 절차
1. 실행 코드 수정·테스트 완료 → **검증된 코드 커밋**(태그 `mobilens-seongnam-final-code`).
2. 그 커밋의 깨끗한 작업트리에서 `python run_pipeline.py` 실행 → `outputs/run_metadata.json`의 `git_sha`가 코드 커밋을 가리키고
   `code_modified_since_commit=false`(src·스크립트·대시보드·requirements·원본 데이터 기준).
3. 산출물·검증 보고서만 담은 **보고 커밋**(태그 `mobilens-seongnam-final-submission`) — 실행 로직 변경 없음.
   따라서 최종 HEAD(보고 커밋)에서 `git_sha`가 부모(코드 커밋)를 가리키는 것이 정상이다. 커밋이 자기 SHA를 담을 수는 없다.
4. 제출 ZIP(`scripts/make_submission_zip.py`)은 git 메타데이터 없이 동작하며, 이 경우 메타데이터의 `git_sha`는 null이 된다.

## 코드 구조
```
run_pipeline.py  demo.py  dashboard/app.py
src/   config · data_validation · preprocessing · geocoding · network · accessibility · mvi · spatial_statistics
       sensitivity · hotspot · optimization · baseline · kpi · routing · policy_scenarios · what_if
       explanation · grounding_validator · visualization · reporting
tests/ scripts/(preflight · make_submission_zip · compare_outputs · download_osm_network · llm_grounding_check)
docs/  data/raw  data/processed  outputs
```
