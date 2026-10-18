"""OpenStreetMap 도로망 스냅샷 다운로드 (선택 실행 — 인터넷 필요, 파이프라인은 저장된 스냅샷만 사용).

사용: python scripts/download_osm_network.py
필요: pip install -r requirements-optional.txt  (osmnx)

성남시 행정동 경계 합집합에 BUFFER_DEG(약 2km) 여유를 둔 영역의 보행망(walk)·차량망(drive)을
받아, 최단경로 계산에 필요한 정보(노드 좌표, 방향 간선 길이)만 data/raw/osm/ 에 저장한다.
OSM은 계속 갱신되므로 재다운로드 시 결과가 달라질 수 있다 — 분석 재현은 저장된 스냅샷으로 한다.
데이터 출처: © OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

BUFFER_DEG = 0.02
GZ = {"method": "gzip", "mtime": 0}  # 결정적 gzip (재저장 시 바이트 동일)


def main() -> int:
    import osmnx as ox
    import tempfile
    ox.settings.cache_folder = tempfile.mkdtemp(prefix="osmnx_")  # Overpass 응답 캐시는 저장소 밖에
    import geopandas as gpd
    import pandas as pd
    from src import config as C

    out_dir = C.OSM_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    poly = gpd.read_file(C.RAW_BOUNDARY).union_all().buffer(BUFFER_DEG)
    meta = {"source": "OpenStreetMap via Overpass API (osmnx)",
            "license": "© OpenStreetMap contributors, ODbL 1.0",
            "downloaded_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "osmnx_version": ox.__version__, "area": f"성남시 행정동 경계 합집합 + {BUFFER_DEG}° buffer",
            "networks": {}}
    for nt in ["walk", "drive"]:
        G = ox.graph_from_polygon(poly, network_type=nt, simplify=True, retain_all=False)
        nodes = pd.DataFrame([(n, round(d["x"], 7), round(d["y"], 7)) for n, d in G.nodes(data=True)],
                             columns=["node", "lon", "lat"]).sort_values("node")
        edges = pd.DataFrame([(u, v, d["length"]) for u, v, d in G.edges(data=True)],
                             columns=["u", "v", "length_m"])
        edges = edges.groupby(["u", "v"], as_index=False)["length_m"].min()
        edges["length_m"] = edges["length_m"].round(2)
        edges = edges.sort_values(["u", "v"])
        nodes.to_csv(out_dir / f"seongnam_{nt}_nodes.csv.gz", index=False, compression=GZ)
        edges.to_csv(out_dir / f"seongnam_{nt}_edges.csv.gz", index=False, compression=GZ)
        meta["networks"][nt] = {"n_nodes": len(nodes), "n_directed_edges": len(edges),
                                "nodes_file": f"seongnam_{nt}_nodes.csv.gz",
                                "edges_file": f"seongnam_{nt}_edges.csv.gz"}
        print(f"{nt}: nodes={len(nodes)} edges={len(edges)}")
    (out_dir / "osm_snapshot_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
