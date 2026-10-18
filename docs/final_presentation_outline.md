# 최종 발표 흐름 (5분) — 모든 숫자는 outputs/presentation/12_key_numbers.json

| 시간 | 장면 | 자료 |
|---|---|---|
| 0:00–0:30 | 문제: 고령 177,033명, 가까움 ≠ 이용 가능 (최근접 vs E2SFCA ρ=0.23) | `01_problem_summary.json` |
| 0:30–1:10 | 데이터→접근성: 의료·복지 E2SFCA(OSM 보행망), 대중교통 | 대시보드 [이동취약성] |
| 1:10–1:50 | MVI = Need × (1−Access), 상위 동 원인 분해 | `03_mvi_map.png`, `02_top_vulnerable_regions.csv` |
| 1:50–2:30 | Hotspot: 36 시나리오에서 남는 안정 4개 동 vs 불안정 | `04_hotspot_map.png`, `10_robustness_summary.json` |
| 2:30–3:20 | 최적화: MCLP +24.4%p, 정책 A/B/C 동일 KPI·노선 | `05_mclp_map.png`, `06`, `07_policy_tradeoff.png` |
| 3:20–4:00 | What-if 실시간 재계산 (k·반경·차량) | 대시보드 [What-if] |
| 4:00–4:40 | 설명 + grounding: 정상 PASS / 변조 5종 FAIL | 대시보드 [설명·검증], `11_grounding_validation.json` |
| 4:40–5:00 | 한계·재현성: proxy, 가정값, preflight READY | `docs/limitations.md` |

핵심 메시지: **탐지(E2SFCA·MVI) → 검증(LISA·강건성) → 대안(입지·노선 최적화) → 재평가(동일 KPI·What-if) → 설명검증**.
