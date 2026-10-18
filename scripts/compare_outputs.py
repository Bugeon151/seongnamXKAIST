"""기존 산출물(저장소)과 재생성 산출물(MOBILENS_OUT_ROOT)을 파일 단위로 비교한다.

사용: python scripts/compare_outputs.py outputs_validation
판정:
  IDENTICAL        줄바꿈(CRLF/LF) 차이를 제외하고 바이트 동일
  SAME_VALUES      CSV/JSON 값이 모두 동일 (숫자 표기·키 순서 등 서식만 다름)
  NUMERIC_TOL      숫자 차이가 상대 1e-6 이내 (부동소수 반올림 수준)
  DIFFERENT        값 차이 존재 — 상세 예시를 함께 기록
  BINARY_DIFF      이미지/HTML 등 렌더링 산출물 (값 비교 대상 아님)
  VOLATILE         실행 시각·소요시간·환경 정보만 담은 파일 (run_metadata.json, pipeline_status.csv)
결과는 <root>/COMPARISON.md 에 저장한다.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DIRS = ["data/processed", "outputs"]
RENDERED = {".png", ".html", ".jpg", ".svg"}
VOLATILE = {"outputs/run_metadata.json", "outputs/logs/pipeline_status.csv"}


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _cmp_scalar(a, b, diffs, where, tol_hits):
    if a == b:
        return
    fa, fb = _num(a), _num(b)
    if fa is not None and fb is not None:
        if fa == fb:
            return
        if abs(fa - fb) <= 1e-6 * max(1.0, abs(fa), abs(fb)):
            tol_hits.append(where)
            return
    if (a in ("", None) or (isinstance(a, float) and math.isnan(a))) and \
       (b in ("", None) or (isinstance(b, float) and math.isnan(b))):
        return
    diffs.append(f"{where}: {a!r} → {b!r}")


def _cmp_json(a, b, diffs, where, tol_hits):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                diffs.append(f"{where}.{k}: {'없음' if k not in a else '있음'} → {'없음' if k not in b else '있음'}")
            else:
                _cmp_json(a[k], b[k], diffs, f"{where}.{k}", tol_hits)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{where}: 길이 {len(a)} → {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _cmp_json(x, y, diffs, f"{where}[{i}]", tol_hits)
    else:
        _cmp_scalar(a, b, diffs, where, tol_hits)


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def compare(old: Path, new: Path) -> tuple[str, list[str]]:
    ob, nb = old.read_bytes().replace(b"\r\n", b"\n"), new.read_bytes().replace(b"\r\n", b"\n")
    if ob == nb:
        return "IDENTICAL", []
    suf = old.suffix.lower()
    if suf in RENDERED:
        return "BINARY_DIFF", [f"{len(ob)} → {len(nb)} bytes"]
    diffs: list[str] = []
    tol: list[str] = []
    if suf == ".csv":
        a = pd.read_csv(old, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        b = pd.read_csv(new, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if list(a.columns) != list(b.columns):
            diffs.append(f"컬럼 차이: -{sorted(set(a.columns) - set(b.columns))} "
                         f"+{sorted(set(b.columns) - set(a.columns))}")
        if len(a) != len(b):
            diffs.append(f"행 수 {len(a)} → {len(b)}")
        for c in [c for c in a.columns if c in b.columns]:
            for i, (x, y) in enumerate(zip(a[c], b[c])):
                _cmp_scalar(x, y, diffs, f"row{i + 1}.{c}", tol)
    elif suf in {".json", ".geojson"}:
        try:
            _cmp_json(_load_json(old), _load_json(new), diffs, "$", tol)
        except json.JSONDecodeError:
            return "DIFFERENT", ["JSON 파싱 불가"]
    else:
        ol, nl = ob.decode("utf-8", "replace").splitlines(), nb.decode("utf-8", "replace").splitlines()
        diffs = [f"line{i + 1}: {x!r} → {y!r}" for i, (x, y) in enumerate(zip(ol, nl)) if x != y]
        if len(ol) != len(nl):
            diffs.append(f"줄 수 {len(ol)} → {len(nl)}")
    if diffs:
        return "DIFFERENT", diffs
    return ("NUMERIC_TOL", tol[:5]) if tol else ("SAME_VALUES", [])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = (BASE / argv[0]).resolve() if argv else BASE / "outputs_validation"
    shown = root.relative_to(BASE) if root.is_relative_to(BASE) else root
    lines = [f"# 산출물 비교: 저장소 기존 결과 vs `{shown}` 재생성 결과", "",
             "| 판정 | 파일 | 상세 (최대 5건) |", "|---|---|---|"]
    counts: dict[str, int] = {}
    for d in DIRS:
        olds = {p.relative_to(BASE / d) for p in (BASE / d).rglob("*") if p.is_file()}
        news = {p.relative_to(root / d) for p in (root / d).rglob("*") if p.is_file()} \
            if (root / d).exists() else set()
        for rel in sorted(olds | news, key=str):
            name = f"{d}/{rel.as_posix()}"
            if rel not in news:
                verdict, det = "NOT_REGENERATED", ["재생성 파이프라인이 만들지 않는 파일"]
            elif rel not in olds:
                verdict, det = "NEW_ONLY", ["기존 저장소에 없던 파일"]
            elif name in VOLATILE:
                verdict, det = "VOLATILE", ["시각·소요시간·환경 정보"]
            else:
                verdict, det = compare(BASE / d / rel, root / d / rel)
            counts[verdict] = counts.get(verdict, 0) + 1
            n_more = f" 외 {len(det) - 5}건" if len(det) > 5 else ""
            detail = "; ".join(det[:5]).replace("|", "\\|") + n_more
            lines.append(f"| {verdict} | `{name}` | {detail} |")
    lines[1:1] = ["", "요약: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), ""]
    out = root / "COMPARISON.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
