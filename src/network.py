"""도로망 최단거리 — data/raw/osm 스냅샷(OSM, ODbL)으로 오프라인·결정적 계산.

- walk : 보행망. 동 중심점 → 시설 보행 네트워크 거리 (접근성 E2SFCA·최근접 거리)
- drive: 차량망(일방통행 방향 반영). 이동지원 차량 노선 거리 (정책 KPI: vehicle-km)

점(중심점·시설·정류장)은 가장 가까운 네트워크 노드에 스냅하며, 스냅 직선거리를 경로 양끝에
더한다(off-network 구간). 단위는 모두 미터(m).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from src import config as C

_TO_METRIC = Transformer.from_crs(C.CRS_WGS84, C.CRS_METRIC, always_xy=True)
_TO_WGS84 = Transformer.from_crs(C.CRS_METRIC, C.CRS_WGS84, always_xy=True)


class Network:
    """OSM 스냅샷 한 종류(walk/drive)의 희소 그래프 + 노드 KD-tree."""

    def __init__(self, kind: str):
        nodes_p = C.OSM_DIR / f"seongnam_{kind}_nodes.csv.gz"
        edges_p = C.OSM_DIR / f"seongnam_{kind}_edges.csv.gz"
        if not nodes_p.exists() or not edges_p.exists():
            raise FileNotFoundError(
                f"OSM 도로망 스냅샷 없음: {nodes_p.name}, {edges_p.name} — "
                "python scripts/download_osm_network.py 로 생성 (인터넷 필요)")
        nodes = pd.read_csv(nodes_p)
        edges = pd.read_csv(edges_p)
        self.kind = kind
        self.node_ids = nodes["node"].to_numpy()
        idx = pd.Series(np.arange(len(nodes)), index=self.node_ids)
        u = idx.loc[edges["u"].to_numpy()].to_numpy()
        v = idx.loc[edges["v"].to_numpy()].to_numpy()
        n = len(nodes)
        self.graph = csr_matrix((edges["length_m"].to_numpy(float), (u, v)), shape=(n, n))
        self.lonlat = (nodes["lon"].to_numpy(float), nodes["lat"].to_numpy(float))
        x, y = _TO_METRIC.transform(*self.lonlat)
        self.xy = np.column_stack([x, y])
        self.directed = kind == "drive"  # 보행망은 양방향으로 취급
        # 스냅 대상: 차량망은 강연결 최대요소(모든 쌍 상호 도달 가능), 보행망은 약연결 최대요소
        _, lab = connected_components(self.graph, directed=True,
                                      connection="strong" if self.directed else "weak")
        main = np.bincount(lab).argmax()
        self.snap_nodes = np.flatnonzero(lab == main)
        self.tree = cKDTree(self.xy[self.snap_nodes])

    def snap(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """경위도 점 → (최근접 노드 인덱스, 스냅 직선거리 m)."""
        x, y = _TO_METRIC.transform(np.asarray(lon, float), np.asarray(lat, float))
        d, i = self.tree.query(np.column_stack([x, y]))
        return self.snap_nodes[i], d

    def distance_matrix(self, src_lon, src_lat, dst_lon, dst_lat,
                        limit: float = np.inf) -> np.ndarray:
        """출발점×도착점 네트워크 거리(m). limit 초과·도달불가는 inf.

        반환값 = 출발 스냅거리 + 노드간 최단경로 + 도착 스냅거리.
        """
        si, sd = self.snap(src_lon, src_lat)
        di, dd = self.snap(dst_lon, dst_lat)
        uniq, inv = np.unique(si, return_inverse=True)
        lim = limit if np.isinf(limit) else max(0.0, limit)
        dist = dijkstra(self.graph, directed=self.directed, indices=uniq, limit=lim)
        m = dist[inv][:, di] + sd[:, None] + dd[None, :]
        if not np.isinf(limit):
            m[m > limit] = np.inf
        return m


@lru_cache(maxsize=2)
def load(kind: str) -> Network:
    return Network(kind)


def demand_points(dong_gdf) -> pd.DataFrame:
    """행정동 수요 대표점 — 모든 모듈(접근성·MCLP 커버·보행거리)이 같은 점을 쓴다.

    대표점 = 동 내 버스정류장(행정동 매칭분)의 medoid: 같은 동 다른 정류장들까지 직선거리 합이 최소인 정류장.
    정류장은 사람이 사는 도로를 따라 놓이므로, 산지·공원을 넓게 포함한 동에서 폴리곤 중심점이 거주지와
    떨어지는 문제(예: 중심점이 산지 산책로에 스냅)를 피한다. 정류장이 없는 동은 폴리곤 내부 대표점.
    반환: dong_code, lon, lat, method, centroid_lon, centroid_lat, shift_m(중심점과의 거리)
    """
    bus = pd.read_csv(C.BUS_CLEAN, encoding="utf-8-sig")
    bus = bus[bus["join_status"] == "matched"].copy()
    bus["official_dong_code"] = bus["official_dong_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    dm = dong_gdf.to_crs(C.CRS_METRIC)
    cent = dm.geometry.centroid
    rows = []
    for code, geom_c, geom in zip(dong_gdf["dong_code"].astype(str), cent, dm.geometry):
        st = bus[bus["official_dong_code"] == code]
        if len(st):
            x, y = _TO_METRIC.transform(st["경도"].to_numpy(float), st["위도"].to_numpy(float))
            xy = np.column_stack([x, y])
            i = int(np.argmin(np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(2)).sum(1)))
            lon, lat, method, px, py = float(st["경도"].iloc[i]), float(st["위도"].iloc[i]), "bus_stop_medoid", x[i], y[i]
        else:
            rp = geom.representative_point()
            lon, lat = _TO_WGS84.transform(rp.x, rp.y)
            method, px, py = "polygon_representative_point", rp.x, rp.y
        clon, clat = _TO_WGS84.transform(geom_c.x, geom_c.y)
        rows.append({"dong_code": code, "lon": lon, "lat": lat, "method": method, "centroid_lon": clon,
                     "centroid_lat": clat, "shift_m": round(float(np.hypot(px - geom_c.x, py - geom_c.y)), 1)})
    return pd.DataFrame(rows)


def dong_centroids_lonlat(dong_gdf) -> tuple[np.ndarray, np.ndarray]:
    """행정동 수요 대표점(경위도). 이름은 하위호환 — 실제 값은 demand_points()."""
    dp = demand_points(dong_gdf)
    return dp["lon"].to_numpy(), dp["lat"].to_numpy()
