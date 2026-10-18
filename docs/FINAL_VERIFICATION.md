# 최종 제출 검증 기록 (FINAL_VERIFICATION)

2026 제2회 성남 × KAIST AI 경진대회 본선 제출본 **MobiLens Seongnam**의 릴리스 검증 결과.
수치는 모두 실제 실행 산출물에서 옮겼다(`outputs/presentation/12_key_numbers.json`).

## 1. 커밋과 provenance

| 구분 | 커밋 | 내용 |
|---|---|---|
| 검증 시작 기준 | `c7dc764` | 원격 `seongnam-kaist-ai-2026` HEAD |
| **검증된 코드 커밋** | `b555554` (태그 `mobilens-seongnam-final-code`) | 실행 코드·테스트·문서 수정 |
| **보고 커밋** | 이 문서를 포함한 커밋 (태그 `mobilens-seongnam-final-submission`) | 재생성 산출물·검증 기록만, 실행 로직 변경 없음 |

- `outputs/run_metadata.json`: `git_sha = b555554…`, `code_modified_since_commit = false`, `modified_paths = []`.
  코드 커밋의 깨끗한 작업트리에서 `python run_pipeline.py`를 실행한 결과다.
- 최종 HEAD(보고 커밋)에서 `git_sha`가 부모(코드 커밋)를 가리키는 것은 정상이다. 커밋은 자기 SHA를 담을 수 없으므로
  '코드 커밋에서 실행 → 산출물만 담은 보고 커밋' 순서를 쓴다. 보고 커밋은 `src/`, `run_pipeline.py`, `demo.py`,
  `dashboard/`, `scripts/`, `tests/`, `requirements*.txt`, `data/raw/`, `data/SHA256SUMS`를 바꾸지 않는다
  (`git diff --stat b555554 HEAD`로 확인 가능).
- 이전 산출물의 `git_sha = 4ac3855…`, `code_modified_since_commit = true`(과거 미커밋 상태에서 실행된 기록)는 이번 재실행으로 해소했다.
- 제출 ZIP은 git 메타데이터 없이 동작하며, ZIP에서 다시 실행하면 `git_sha = null`(정상)로 기록된다.

검증 환경: Windows 11 Home 10.0.26200 (스마트 앱 컨트롤 켜짐), Python 3.13.15, `requirements.txt` 고정 버전.

## 2. 감사 결과와 조치 (P0 0건 · P1 5건 모두 해결)

| 등급 | 문제 | 조치 |
|---|---|---|
| P1 | **정책 C 축약 후보가 정책 A의 해를 포함하지 않음**(분당구 후보 14개 중 8개만 유지). 문서·코드 주석의 "A의 해 항상 포함" 주장이 사실과 다르고, `C ≤ A` 테스트는 우연히 통과 | A의 해를 같은 Held-Karp 거리로 평가해 비교 후보에 추가 → `C ≤ A` 구조적 보장. `policy_results.json`에 `A_solution_in_reduced_candidates=false`, `A_solution_route_km_exact=21.471` 기록. 결과값 불변(17.297km < 21.471km). A/B를 강제로 후보에 넣으면 C가 18.109km로 오히려 나빠짐을 확인하고 축약 규칙은 유지 |
| P1 | **grounding validator 우회 가능**: `12.5km`처럼 단위가 붙은 숫자 미추출(운행거리 변조 통과), `때문입니다`·`탓`·`기인` 통과, 라벨 없는 `80%`가 아무 0~1 값의 반올림과 일치해 통과, `차량 5대`가 거점 수 5와 일치해 통과, 다른 정책 값 끼워넣기 통과, 한글에 붙은 숫자·문장 끝 숫자 미추출 | 지역명 마스킹 후 숫자 추출, `%`/`km`/`분`/`대` 단위 숫자의 KPI 라벨 결합 필수, 정책 비교 값 정책별 대조, 인과·권고 표현·범위 밖 지역 탐지. 실제 설명 53/53 PASS 유지, 적대적 사례 38/38 의도대로 판정 → `tests/test_grounding.py` 고정 |
| P1 | provenance 불일치(`run_metadata` SHA가 HEAD와 다르고 dirty=true), dirty 판정 범위가 `src`·`run_pipeline.py`뿐 | 릴리스 절차로 재실행, dirty 범위를 scripts·dashboard·requirements·원본 데이터까지 확대, 다른 git 저장소 안에 압축해제 시 그 저장소 SHA를 기록하지 않도록 수정, 한글 경로 git 출력 디코딩(cp949) 오류 수정 |
| P1 | 문서·대시보드·산출물 서술이 구현과 불일치: 수요점 "행정동 중심점"(실제는 동 내 정류장 medoid), 정책 C "MILP로 푼다"(실제는 완전탐색) | methodology·limitations·judge_qa·ai_usage·대시보드 문구·`accessibility_summary.json` 정정 |
| P1 | Windows 11 스마트 앱 컨트롤 환경에서 실행 불가: scikit-learn 1.9.1·pyogrio 0.13.0·pyarrow 25(미고정 전이 의존성)의 DLL이 차단되어 esda/libpysal·GeoJSON 입출력·대시보드 import 실패 | 차단되지 않는 scikit-learn 1.8.0·pyogrio 0.12.1·pyarrow 24.0.0으로 고정, 산출물 78개 파일 동일 확인 |
| P2 | 원본 무수정 원칙이 주장만 있고 검사 없음 | `data/SHA256SUMS`(줄바꿈 정규화 SHA-256, OS·git autocrlf 무관) 대조를 데이터 검증 1단계에 추가 |
| P2 | ZIP 제외 규칙 보강(`.env.*`, `cache/`, `build/`), `compare_outputs` 경로 처리, 미사용 import | 수정 |

남긴 P3(변경 위험 대비 이득 작음): 정류장 ID가 `5222.0`처럼 실수형 문자열로 표기됨, `src/geocoding.py`(외부 API 전용, 기본 경로 밖)의 미사용 지역변수.

## 3. 검증 결과

| 항목 | 결과 |
|---|---|
| pytest | **105/105 PASS** (기존 66 + 추가 39: 적대적 grounding, 실데이터 MCLP 완전탐색 대조(k=1,2), 노선 순환시간 재계산·순열 완전탐색 TSP 대조, 정책 C 완전탐색 독립 브루트포스 대조, 실패 주입) |
| 파이프라인 | **8/8 PASS** (약 11초) |
| `--validate` 재현성 | IDENTICAL 78, VOLATILE 2(실행 시각·환경), NOT_REGENERATED 10(외부 API 지오코딩 기록·선택적 LLM 측정 — 파이프라인 입력/기록물), BINARY_DIFF 1(`mvi_interactive.html`: folium 무작위 요소 ID만 다름, 32자리 ID 마스킹 후 동일). **REAL RESULT DIFFERENCE 0** |
| 수정 전 대비 산출물 변화 | 분석 수치 변화 없음. 변경은 라벨·메타데이터뿐: grounding 결합 라벨(`PASS(src)`→`PASS(kpi:…)`), 인과 위반 문구, 정책 C solver 기록 필드 추가, `accessibility_summary.json` 서술·키 이름, 데이터 검증표에 원본 해시 검사 1행 추가 |
| `--demo --tamper` | PASS — 정상 설명 PASS, 숫자·지역·정책·인과·지표 변조 5종 모두 FAIL |
| What-if | `policy=A k=7`, `policy=B k=7 radius_m=1000` 실제 재계산 Optimal |
| 대시보드 | AppTest(전체 탭 렌더·What-if 위젯 재계산) PASS, headless 부팅 health=ok |
| preflight `--package` | **Gate A~J 모두 PASS** → `MOBILENS FINAL RELEASE READY` |
| 보안·개인정보 | 비밀키·토큰·개인키·`.env`·개인 절대경로·개인 이메일 없음. 원자료 기관 대표 연락처는 분석에 미사용(NOTICE) |

### 제출 ZIP 독립 검증 (git 없는 새 폴더 + 새 가상환경)
`scripts/make_submission_zip.py`로 만든 ZIP(150개 파일)을 git 저장소가 아닌 임시 폴더에 풀고, 새 venv에
패키지된 `requirements.txt`만으로 설치(`pip check` 이상 없음) 후 실행:
import PASS · 필수 파일(README·requirements·원본·OSM·지오코딩 입력·SHA256SUMS) 존재 · pytest 105/105 ·
`run_pipeline.py` 8/8 · `--validate` IDENTICAL 78(실제 차이 0) · `--demo --tamper` PASS · 대시보드 부팅 PASS ·
`git_sha = null`(git 없이 정상 동작) · `.env`·비밀정보 없음. 외부 API(지오코딩·OSM 다운로드·LLM)는 호출하지 않는다.
(ZIP 파일 자체의 SHA-256은 ZIP 안에 담을 수 없으므로 릴리스 태그 메시지에 기록한다.)

## 4. 핵심 수치 회귀 (기준값 대비)

| 항목 | 기준 | 실제 | 판정 |
|---|---|---|---|
| 65세 이상 인구 | 177,033명 · 19.67% (2026-08) | 177,033명 · 19.67% | 일치 |
| MVI 1위 | 성남동 ≈ 0.6876 | 성남동 0.6876 | 일치 |
| Global Moran's I | ≈ 0.273, p ≈ 0.004 | 0.2733, p_sim 0.004 | 일치 |
| 안정 Robust Hotspot | 4개(금광2동·상대원1동·수진1동·은행2동) | 동일 4개 | 일치 |
| 강건성 | 36 시나리오, 최소 ρ ≈ 0.8148 | 36, 0.8148 | 일치 |
| 최근접 vs E2SFCA | ρ ≈ 0.2261 | 0.2261 | 일치 |
| MCLP k=5 | 0.8653, top-demand 대비 +24.44%p, greedy 대비 +0.77%p | 0.8653, +24.44, +0.77 | 일치 |
| 정책 A | 0.8653 / 최소 구 0.7745 / 21.47km / 0.403 | 동일 | 일치 |
| 정책 B | 0.8576 / 0.8271 / 20.13km / 0.426 | 동일 | 일치 |
| 정책 C | 0.8714 / 0.7534 / 17.3km / 0.5037 | 동일 | 일치 |
| grounding | 53/53 | 53/53 | 일치 |

## 5. 해석상 유의점 (숨기지 않는 한계)
- 정책 C는 **축약 후보(구별 8개) 내 정확해**이며 전체 1,301개 정류장 전역최적이 아니다. 보장되는 것은
  '정책 A 이상의 취약수요 커버를 A 이하의 (구별 단일 순환) 운행거리로 달성'이다.
- Local Moran p < 0.05는 다중검정 보정을 하지 않았다. 36 시나리오 중 80% 이상 유지되는 동만 '안정'으로 본다.
- 이전에 보고된 "적대적 grounding stress 30/30"은 저장소에 재현 코드가 없었다. 이번에 적대적 사례를
  `tests/test_grounding.py`에 테스트로 고정했고(그 과정에서 위 P1 우회 경로를 발견·수정), 이 수치로 대체한다.
- 시스템은 정책을 결정하지 않는다. 동일 KPI로 대안 간 trade-off를 제시하고, 선택은 사람이 한다.
