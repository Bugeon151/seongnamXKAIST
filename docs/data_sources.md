# 데이터 출처 (data_sources)

모든 원본은 `data/raw/`에 읽기 전용으로 보관한다. 기준 해시는 `data/SHA256SUMS`(줄바꿈 정규화 SHA-256)에 있고 매 실행
데이터 검증 1단계에서 대조한다(불일치 시 FAIL). 파일별 SHA-256·기준일은 `outputs/run_metadata.json`에도 기록된다.

| 데이터 | 제공 | 기준일 | 파일 | 사용 |
|---|---|---|---|---|
| 고령인구현황 | 성남시 공공데이터 | 2023-09~2026-08 (분석 2026년 08월) | `성남시_고령인구현황_202309_202608_UTF8.csv` | Need(고령인구·비율), E2SFCA 수요 |
| 버스정류장 현황 | 성남시 공공데이터 | 2026-04-08 | `경기도 성남시_버스정류장_현황_20260408.csv` | 대중교통 접근성, MCLP 후보지 |
| 의료기관 현황 | 성남시 공공데이터 | 2026-03-04 | `경기도 성남시_의료기관_현황_20260304.csv` | 의료 E2SFCA 공급, 차량 거점(구 보건소) |
| 재가장기요양기관 현황 | 성남시 공공데이터 | 2026-06-05 | `경기도 성남시_재가장기요양기관_현황_20260605.csv` | 복지 E2SFCA 공급(시설이용형 정원) |
| 행정구역 현황 | 성남시 공공데이터 | 2026-05-15 | `경기도 성남시_행정구역_현황_20260515.csv` | 면적, 법정동 어휘(grounding) |
| 행정동 경계 | SGIS 원자료 기반 가공(vuski/admdongkor) | 2026-07-01 | `hangjeongdong_seongnam.geojson` | 50개 동 폴리곤, Queen 가중치 |
| 도로망 | © OpenStreetMap contributors (ODbL) | `osm/osm_snapshot_meta.json` | `osm/seongnam_{walk,drive}_*.csv.gz` | 보행 네트워크 거리, 차량 노선 거리 |

## 외부 지오코딩 (Nominatim, 키 불필요)
- `python -m src.geocoding` / `bounded` / `care` / `depots` 로 실행. 초당 1건 이하, 식별 User-Agent(연락처는
  `NOMINATIM_CONTACT` 환경변수로 지정 가능). 결과는 `data/processed/*_geocoded.csv`로 버전관리하며
  파이프라인은 재조회하지 않고 ID·주소 일치를 검증한 뒤 사용한다.
- Kakao/Naver/VWorld 등 상용 지오코딩 API 키는 사용하지 않는다.

## LLM (선택)
- 로컬 Ollama(`OLLAMA_MODEL`, `OLLAMA_HOST`)만 지원. 외부 API 키 불필요. 없으면 템플릿 설명.
