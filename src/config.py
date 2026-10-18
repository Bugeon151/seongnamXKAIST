"""MobiLens Seongnam — 전역 설정 (단일 source of truth).

모든 경로·파라미터는 여기서만 정의하고, 각 모듈은 config를 import해서 사용한다.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 선택 설정(.env): 이미 설정된 환경변수는 덮어쓰지 않는다
from dotenv import load_dotenv  # noqa: E402
load_dotenv(BASE_DIR / ".env", override=False)

# 산출물 루트 — 기본은 저장소 루트. 환경변수 MOBILENS_OUT_ROOT를 주면
# data/processed·outputs를 그 아래에 새로 생성한다(기존 결과 무수정 재현 검증용).
_out = os.environ.get("MOBILENS_OUT_ROOT", "").strip()
OUT_ROOT = (BASE_DIR / _out).resolve() if _out else BASE_DIR

# --- 원본 데이터 (읽기 전용, 절대 수정 금지) ---
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_POPULATION = RAW_DIR / "성남시_고령인구현황_202309_202608_UTF8.csv"
RAW_BUS = RAW_DIR / "경기도 성남시_버스정류장_현황_20260408.csv"
RAW_MEDICAL = RAW_DIR / "경기도 성남시_의료기관_현황_20260304.csv"
RAW_CARE = RAW_DIR / "경기도 성남시_재가장기요양기관_현황_20260605.csv"
RAW_AREA = RAW_DIR / "경기도 성남시_행정구역_현황_20260515.csv"
RAW_BOUNDARY = RAW_DIR / "hangjeongdong_seongnam.geojson"
# OpenStreetMap 도로망 스냅샷 (scripts/download_osm_network.py로 생성, 이후 읽기 전용)
OSM_DIR = RAW_DIR / "osm"

# --- 가공 데이터 ---
PROCESSED_DIR = OUT_ROOT / "data" / "processed"
INTERIM_DIR = OUT_ROOT / "data" / "interim"
POPULATION_CLEAN = PROCESSED_DIR / "population_clean.csv"
BUS_CLEAN = PROCESSED_DIR / "bus_stops_clean.csv"
MEDICAL_CLEAN = PROCESSED_DIR / "medical_facilities_clean.csv"
DONG_FEATURES = PROCESSED_DIR / "dong_features_base.csv"
DONG_MASTER = PROCESSED_DIR / "seongnam_dong_master.geojson"
# 지오코딩 결과는 외부 API(Nominatim) 실측값이라 재실행마다 달라질 수 있으므로
# 버전관리되는 입력 산출물로 취급한다 — 항상 저장소의 data/processed에서 읽는다.
MEDICAL_GEOCODED = BASE_DIR / "data" / "processed" / "medical_facilities_geocoded.csv"
CARE_GEOCODED = BASE_DIR / "data" / "processed" / "care_facilities_geocoded.csv"
DEPOT_GEOCODED = BASE_DIR / "data" / "processed" / "depots_geocoded.csv"  # 의료 지오코딩 실패 보건소 보완
CARE_CLEAN = PROCESSED_DIR / "care_facilities_clean.csv"
GEOCODE_TARGETS = OUT_ROOT / "outputs" / "geocoding_targets_medical.csv"
GEOCODE_TARGETS_CARE = OUT_ROOT / "outputs" / "geocoding_targets_care.csv"

# --- 산출물 ---
OUTPUTS = OUT_ROOT / "outputs"
MAPS = OUTPUTS / "maps"
TABLES = OUTPUTS / "tables"
JSON_DIR = OUTPUTS / "json"
FIGURES = OUTPUTS / "figures"
LOGS = OUTPUTS / "logs"

# --- 분석 파라미터 ---
CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:5186"  # 한국 중부원점 (거리 계산용)
KOREA_LAT = (33.0, 39.0)
KOREA_LON = (124.0, 132.0)

N_DONG = 50  # 분석 대상 행정동 수 (고정 검증값)
BASE_MONTH = "2026년08월"  # 인구 기준월 (데이터 내 최신월)

# --- 접근성 (src/accessibility.py) ---
# E2SFCA 거리 밴드 (하한 m, 상한 m, 가중치) — OSM 보행망 네트워크 거리 기준
E2SFCA_BANDS = [(0, 1000, 1.0), (1000, 2000, 0.6), (2000, 3000, 0.25)]
NEAR_THRESHOLD_M = 1000  # '1km 내 시설 수' 기준 반경
DECAY_SCALE_VARIANTS = [0.75, 1.0, 1.25]  # 거리감쇠 민감도: 밴드 경계 배율

# --- MVI (src/mvi.py) ---
# Need  = mean(minmax(65세 이상 인구), minmax(고령인구 비율))
# Access = mean(minmax(의료 E2SFCA), minmax(복지 E2SFCA), minmax(고령 1천명당 정류장))
# MVI   = Need^(2·w_need) × (1 − Access)^(2·w_access)   — 균형(0.5, 0.5)에서 Need × (1 − Access)
ACCESS_COMPONENTS = {
    "medical": "med_e2sfca_network",
    "welfare": "wel_e2sfca_network",
    "transit": "transit_stops_per_1000_elderly",
}
MVI_SCENARIOS = {  # (w_need, w_access)
    "A_balanced": (0.50, 0.50),
    "B_need60": (0.60, 0.40),
    "C_access60": (0.40, 0.60),
}
MVI_PRIMARY = "A_balanced"

# --- 공간통계·강건성 (src/spatial_statistics.py, src/sensitivity.py) ---
MORAN_PERMUTATIONS = 999
MORAN_ALPHA = 0.05
MORAN_SEED = 42
# 강건성 격자: MVI 가중치 × 거리감쇠 × 결합형식(곱/가중합) × 정규화(min-max/순위)
ROBUST_FORMS = ["product", "additive"]
ROBUST_NORMALIZATIONS = ["minmax", "rank"]
ROBUST_STABLE_SHARE = 0.8    # 격자 시나리오의 80% 이상에서 HH·유의 → 안정 Robust Hotspot
ROBUST_UNSTABLE_SHARE = 0.2  # 20% 이상 80% 미만 → 불안정 hotspot (조건부)
TOP_K_VULNERABLE = 10        # '상위 취약지역' 정의 (MVI 상위 k개 동)

# --- MCLP (src/optimization.py) ---
MCLP_K_VALUES = [3, 5, 7]
COVERAGE_RADIUS_M = 1500  # 신규 이동지원 거점(정류장)의 서비스 반경(직선거리)
COVERAGE_CURVE_K = list(range(1, 11))  # 자원 수 대비 커버리지 곡선
RANDOM_BASELINE_SEEDS = 20
RANDOM_SEED = 42

# --- 정책·노선 (src/policy_scenarios.py, src/routing.py) ---
K_MAIN = 5  # 정책 A/B 비교 기준 자원 수
# 차량 거점: 각 구 보건소 (의료기관 원자료의 '보건소' 종별). 정류장은 소속 구 보건소 차량이 운행.
# 아래 값은 운영 가정(assumption)이며 What-if에서 변경 가능하다.
VEHICLE_SPEED_KMH = 20.0      # 도심 평균 운행속도 가정 (신호·정체 포함)
DWELL_MIN_PER_STOP = 3.0      # 정류장당 승하차 정차시간 가정 (분)
VEHICLES_PER_DEPOT = 1        # 구 보건소별 차량 수
MAX_CYCLE_MIN = 90.0          # 차량 1대의 1회 순환 최대 소요시간 (배차간격 상한 가정, 분)
# Policy C (운영효율): 노선 길이 예산(km)별 커버리지 최대화 — Pareto 곡선
POLICY_C_BUDGETS_KM = [10, 15, 20, 25, 30, 40, 50, 60, 80]
POLICY_C_MAX_STOPS = 10

# --- LLM 설명 (src/explanation.py) — 선택 계층, 없어도 파이프라인 동작 ---
OLLAMA_HOST_DEFAULT = "http://localhost:11434"
LLM_MAX_RETRIES = 1           # grounding FAIL 시 위반 내용을 알려주고 재생성하는 횟수

METHOD_VERSION = "mobilens-2.0"
