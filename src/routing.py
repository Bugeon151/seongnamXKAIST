"""이동지원 차량 노선 — 구 보건소 거점, OSM 차량망 거리, 정확해(Held-Karp + 분할 DP).

운영 모형(가정, docs/methodology.md):
  - 각 구 보건소(의료기관 원자료 '보건소' 종별)가 차량 거점(depot)이다.
  - 선정 정류장은 소속 구(행정동 → 구) 보건소 차량이 순환 운행한다.
  - 차량 1대의 1회 순환(거점 출발 → 정류장들 → 거점 복귀) 소요시간
      = 노선거리 / VEHICLE_SPEED_KMH + DWELL_MIN_PER_STOP × 정류장 수   ≤ MAX_CYCLE_MIN
  - 구별 차량 수는 VEHICLES_PER_DEPOT 이상에서 순환시간 제약을 만족하는 최소 대수.
  - 목적: 총 vehicle-km 최소. 구별 정류장 ≤ MAX_EXACT_STOPS 에서 정확해.
거리: OSM 차량망 방향그래프 최단경로(m). 강연결 최대요소 노드에만 스냅하여 도달불가 쌍을 없앤다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.sparse.csgraph import dijkstra

from src import config as C
from src import network as N

MAX_EXACT_STOPS = 12


@dataclass
class RouteParams:
    speed_kmh: float = C.VEHICLE_SPEED_KMH
    dwell_min: float = C.DWELL_MIN_PER_STOP
    vehicles_per_depot: int = C.VEHICLES_PER_DEPOT
    max_cycle_min: float = C.MAX_CYCLE_MIN


@dataclass
class RoutePlan:
    routes: list[dict] = field(default_factory=list)   # 차량별 노선
    total_km: float = 0.0
    max_cycle_min: float = 0.0
    vehicles_used: int = 0
    vehicles_required_by_gu: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"total_vehicle_km": round(self.total_km, 2), "max_cycle_min": round(self.max_cycle_min, 1),
                "vehicles_used": self.vehicles_used, "vehicles_required_by_gu": self.vehicles_required_by_gu,
                "routes": self.routes}


def gu_from_adm(adm_nm_full: str) -> str:
    """'경기도 성남시수정구 신흥1동' → '수정구'."""
    return adm_nm_full.split()[1].replace("성남시", "")


@lru_cache(maxsize=1)
def dong_gu_map() -> dict[str, str]:
    dong = gpd.read_file(C.DONG_MASTER, encoding="utf-8")
    return {str(c): gu_from_adm(a) for c, a in zip(dong["dong_code"], dong["adm_nm_full"])}


def load_depots() -> pd.DataFrame:
    """구 보건소 거점 좌표. 의료기관 지오코딩 결과 우선, 실패분은 depots_geocoded.csv(별도 실측)."""
    med = pd.read_csv(C.MEDICAL_CLEAN, encoding="utf-8-sig")
    hc = med[med["의료기관종별"] == "보건소"].drop_duplicates("unique_facility_id")
    geo = pd.read_csv(C.MEDICAL_GEOCODED, encoding="utf-8-sig").set_index("unique_facility_id")
    extra = (pd.read_csv(C.DEPOT_GEOCODED, encoding="utf-8-sig").set_index("unique_facility_id")
             if C.DEPOT_GEOCODED.exists() else pd.DataFrame())
    rows = []
    for _, r in hc.iterrows():
        fid = r["unique_facility_id"]
        g = geo.loc[fid]
        if g["geocode_status"] == "success" and str(g["in_seongnam"]) == "True":
            lon, lat, src = float(g["longitude"]), float(g["latitude"]), "medical_facilities_geocoded.csv"
        elif fid in extra.index and str(extra.loc[fid, "in_seongnam"]) == "True":
            lon, lat, src = float(extra.loc[fid, "longitude"]), float(extra.loc[fid, "latitude"]), \
                "depots_geocoded.csv"
        else:
            raise FileNotFoundError(f"차량 거점 {r['의료기관명']} 좌표 없음 — python -m src.geocoding depots 실행 필요")
        rows.append({"gu": r["구별"], "depot_name": r["의료기관명"], "unique_facility_id": fid,
                     "lon": lon, "lat": lat, "coord_source": src})
    dep = pd.DataFrame(rows)
    if sorted(dep["gu"]) != ["분당구", "수정구", "중원구"]:
        raise ValueError(f"구별 보건소 3개가 아님: {dep['gu'].tolist()}")
    return dep


def held_karp(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """노드 0=거점, 1..n=정류장. 모든 정류장 부분집합의 최적 순환비용과 역추적 정보.

    반환 (cost[mask], parent) — cost[mask]: mask 정류장만 방문하는 최단 순환(거점→…→거점).
    """
    n = D.shape[0] - 1
    if n > MAX_EXACT_STOPS:
        raise ValueError(f"구별 정류장 {n}개 — 정확해는 {MAX_EXACT_STOPS}개 이하만 지원")
    full = 1 << n
    dp = np.full((full, n), np.inf)
    par = np.full((full, n), -1, dtype=int)
    for j in range(n):
        dp[1 << j, j] = D[0, j + 1]
    for mask in range(1, full):
        for j in range(n):
            if not (mask >> j) & 1 or not np.isfinite(dp[mask, j]):
                continue
            base = dp[mask, j]
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                v = base + D[j + 1, k + 1]
                if v < dp[nm, k]:
                    dp[nm, k] = v
                    par[nm, k] = j
    cost = np.full(full, np.inf)
    cost[0] = 0.0
    for mask in range(1, full):
        tot = dp[mask] + D[1:, 0]
        cost[mask] = tot.min()
    return cost, (dp, par)


def _order(mask: int, D: np.ndarray, hk) -> list[int]:
    dp, par = hk
    tot = dp[mask] + D[1:, 0]
    j = int(np.argmin(tot))
    seq = []
    while j != -1:
        seq.append(j)
        pj = par[mask, j]
        mask ^= 1 << j
        j = pj
    return seq[::-1]


def _partition(cost: np.ndarray, feasible: np.ndarray, n: int, v: int) -> tuple[float, list[int]]:
    """정류장 전체를 최대 v개 노선(각 노선 feasible)으로 분할하는 최소 총거리 (부분집합 DP)."""
    full = (1 << n) - 1
    best = {0: (0.0, [])}
    layer = {0: (0.0, [])}
    for _ in range(v):
        nxt = dict(layer)
        for mask, (c0, parts) in layer.items():
            rest = full ^ mask
            if rest == 0:
                continue
            low = rest & -rest
            sub = rest
            while sub:
                if sub & low and feasible[sub]:
                    nm, c = mask | sub, c0 + cost[sub]
                    if nm not in nxt or c < nxt[nm][0] - 1e-9:
                        nxt[nm] = (c, parts + [sub])
                sub = (sub - 1) & rest
        layer = nxt
        best = layer
    return best.get(full, (np.inf, []))


def plan_routes(stops: pd.DataFrame, params: RouteParams | None = None, with_paths: bool = False) -> RoutePlan:
    """선정 정류장(경도·위도·official_dong_code·정류장명 필요) → 구별 최적 노선."""
    p = params or RouteParams()
    plan = RoutePlan()
    if len(stops) == 0:
        return plan
    dep = load_depots().set_index("gu")
    gmap = dong_gu_map()
    st = stops.copy()
    st["gu"] = st["official_dong_code"].astype(str).str.replace(r"\.0$", "", regex=True).map(gmap)
    if st["gu"].isna().any():
        raise ValueError("정류장의 소속 구를 찾을 수 없음 (official_dong_code 확인)")
    drive = N.load("drive")
    m_per_min = p.speed_kmh * 1000 / 60
    for gu, grp in st.groupby("gu", sort=True):
        d = dep.loc[gu]
        lon = np.r_[d["lon"], grp["경도"].to_numpy(float)]
        lat = np.r_[d["lat"], grp["위도"].to_numpy(float)]
        D = drive.distance_matrix(lon, lat, lon, lat)
        if not np.isfinite(D).all():
            raise RuntimeError(f"{gu} 차량망 도달불가 쌍 존재 — 스냅 확인 필요")
        n = len(grp)
        cost, hk = held_karp(D)
        masks = np.arange(1 << n)
        nstops = np.array([bin(m).count("1") for m in masks])
        minutes = cost / m_per_min + p.dwell_min * nstops
        feasible = minutes <= p.max_cycle_min
        feasible[0] = False
        v = max(1, p.vehicles_per_depot)
        total, parts = _partition(cost, feasible, n, v)
        while not np.isfinite(total) and v < n:
            v += 1
            total, parts = _partition(cost, feasible, n, v)
        if not np.isfinite(total):
            raise RuntimeError(f"{gu}: 정류장 1개 노선도 최대 순환시간 {p.max_cycle_min}분 초과 — 파라미터 확인")
        plan.vehicles_required_by_gu[gu] = len(parts)
        for part in parts:
            order = _order(part, D, hk)
            seq = [0] + [o + 1 for o in order] + [0]
            leg = [float(D[a, b]) for a, b in zip(seq[:-1], seq[1:])]
            r = {"gu": gu, "depot": d["depot_name"],
                 "stops": [str(grp.iloc[o]["정류장명"]) for o in order],
                 "stop_ids": [str(grp.iloc[o]["정류장번호(ID)"]) for o in order],
                 "km": round(sum(leg) / 1000, 3),
                 "cycle_min": round(sum(leg) / m_per_min + p.dwell_min * len(order), 1)}
            if with_paths:
                r["path_lonlat"] = _path_lonlat(drive, lon[seq], lat[seq])
            plan.routes.append(r)
    plan.total_km = sum(r["km"] for r in plan.routes)
    plan.max_cycle_min = max(r["cycle_min"] for r in plan.routes)
    plan.vehicles_used = len(plan.routes)
    return plan


def _path_lonlat(net: N.Network, lon: np.ndarray, lat: np.ndarray) -> list[list[float]]:
    """노선 순서대로 실제 도로 경로 좌표열 (지도 표시용)."""
    idx, _ = net.snap(lon, lat)
    _, pred = dijkstra(net.graph, directed=True, indices=np.unique(idx), return_predecessors=True)
    row = {n: i for i, n in enumerate(np.unique(idx))}
    coords: list[list[float]] = []
    tr_lon, tr_lat = net.lonlat
    for a, b in zip(idx[:-1], idx[1:]):
        seg, cur = [], b
        pr = pred[row[a]]
        while cur != a and cur >= 0:
            seg.append(cur)
            cur = pr[cur]
        seg.append(a)
        for nd in seg[::-1]:
            coords.append([round(float(tr_lon[nd]), 6), round(float(tr_lat[nd]), 6)])
    return coords
