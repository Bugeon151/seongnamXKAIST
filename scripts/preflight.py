"""제출 전 자동 점검 — Acceptance Gate A~J.

사용: python scripts/preflight.py            (A~I: 저장소에서 검사)
      python scripts/preflight.py --package  (+ J: ZIP 생성 → 새 폴더 압축해제 → import·tests·pipeline·demo 재검증)
모든 게이트가 PASS일 때만 'MOBILENS FINAL RELEASE READY' / 'SUBMISSION READY'를 출력한다.

A DATA            원본·가공 데이터 검증 FAIL 0건 (tests/test_data_validation.py 포함)
B REPRODUCIBILITY raw부터 전처리 재현(바이트 동일) + 전체 파이프라인 신규 실행 + 저장소 outputs와 값 비교
C ANALYTICS       접근성·MVI·공간통계·강건성 불변식 테스트
D OPTIMIZATION    MCLP·정책 A/B/C·KPI 일관성 테스트
E VALIDATION      baseline·ablation·Before/After 테스트
F LLM             grounding validator·LLM 흐름(mock) 테스트
G TEST            전체 pytest 통과
H DEMO            CLI 데모 실행 + 대시보드 What-if 테스트
I SECURITY        비밀키·토큰·개인 이메일·절대경로·.env 검사
J PACKAGE         (--package) 제출 ZIP을 새 폴더에서 재검증
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PY = sys.executable
TEXT_SUFFIX = {".py", ".md", ".txt", ".json", ".toml", ".yml", ".yaml", ".cfg", ".ini", ".example", ".csv", ".html"}
SECRET_PATTERNS = {
    "openai_key": r"sk-[A-Za-z0-9]{20,}", "aws_key": r"AKIA[0-9A-Z]{16}", "github_token": r"gh[pousr]_[A-Za-z0-9]{30,}",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "bearer": r"Bearer\s+[A-Za-z0-9._\-]{20,}",
    "assigned_secret": r"(?i)(api[_-]?key|secret|password|token)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{12,}",
}
ABS_PATH = r"(?i)([A-Z]:[\\/]+Users[\\/]+[^\\/\s\"']+|/Users/[A-Za-z][^/\s\"']*|/home/[a-z][^/\s\"']*)"
EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
ALLOWED_EMAILS = {"noreply@anthropic.com"}
SKIP_DIRS = {".git", ".venv", "outputs_validation", "dist", "__pycache__", ".pytest_cache", "interim", "geocode_cache"}
GATE_TESTS = {
    "A": ["test_data_validation.py::", "test_pipeline.py::test_data_validation_no_fail",
          "test_pipeline.py::test_stage_failure"],
    "B": ["test_preprocessing.py::", "test_pipeline.py::test_all_stages_pass"],
    "C": ["test_pipeline.py::test_accessibility_invariants", "test_pipeline.py::test_mvi_invariants",
          "test_pipeline.py::test_spatial_invariants", "test_unit_methods.py::test_e2sfca", "test_unit_methods.py::test_band",
          "test_unit_methods.py::test_normalizations", "test_unit_methods.py::test_combine",
          "test_unit_methods.py::test_components"],
    "D": ["test_pipeline.py::test_mclp_and_baselines", "test_pipeline.py::test_policy", "test_pipeline.py::test_kpi",
          "test_unit_methods.py::test_mclp", "test_unit_methods.py::test_maximin", "test_unit_methods.py::test_pareto",
          "test_unit_methods.py::test_held_karp", "test_unit_methods.py::test_partition",
          "test_unit_methods.py::test_policy_c", "test_pipeline.py::test_routes", "test_pipeline.py::test_mclp_real"],
    "E": ["test_pipeline.py::test_mclp_and_baselines", "test_pipeline.py::test_before_after_arithmetic",
          "test_pipeline.py::test_mvi_invariants"],
    "F": ["test_grounding.py::", "test_pipeline.py::test_grounding_outputs"],
    "H": ["test_pipeline.py::test_what_if", "test_pipeline.py::test_dashboard"],
}


def run(cmd: list[str], cwd: Path, env: dict | None = None, timeout: int = 3600) -> subprocess.CompletedProcess:
    e = dict(os.environ, PYTHONIOENCODING="utf-8", **(env or {}))
    return subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True, encoding="utf-8", timeout=timeout)


def pytest_results(root: Path) -> tuple[bool, dict[str, str], str]:
    xml = root / "outputs_validation" / "pytest_results.xml"
    xml.parent.mkdir(parents=True, exist_ok=True)
    r = run([PY, "-m", "pytest", "tests", "-q", "-p", "no:warnings", "-p", "no:cacheprovider", f"--junitxml={xml}"],
            root)
    res = {}
    for tc in ET.parse(xml).getroot().iter("testcase"):
        name = f"{tc.get('classname').split('.')[-1]}.py::{tc.get('name')}"
        res[name] = "FAIL" if (tc.find("failure") is not None or tc.find("error") is not None) else \
            ("SKIP" if tc.find("skipped") is not None else "PASS")
    return r.returncode == 0, res, r.stdout[-2500:]


def gate_from_tests(res: dict[str, str], prefixes: list[str]) -> tuple[bool, str]:
    sel = {k: v for k, v in res.items() if any(k.startswith(p) for p in prefixes)}
    ok = bool(sel) and all(v == "PASS" for v in sel.values())
    return ok, f"{sum(v == 'PASS' for v in sel.values())}/{len(sel)} tests"


def security_scan(root: Path) -> tuple[bool, list[str]]:
    issues = []
    if (root / ".env").exists():
        issues.append(".env 파일 존재 (제출 ZIP에서는 제외됨 — 내용 확인 필요)")
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if not p.is_file() or any(s in rel.parts for s in SKIP_DIRS) or p.suffix.lower() not in TEXT_SUFFIX:
            continue
        if p.stat().st_size > 5_000_000:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for name, pat in SECRET_PATTERNS.items():
            if re.search(pat, text):
                issues.append(f"{rel}: {name}")
        if p.suffix != ".csv":  # 원자료 CSV의 기관 연락처는 공개 행정정보
            if m := re.search(ABS_PATH, text):
                issues.append(f"{rel}: 절대경로 {m.group(0)[:40]}")
            for em in set(re.findall(EMAIL, text)) - ALLOWED_EMAILS:
                issues.append(f"{rel}: 이메일 {em}")
    return not issues, issues


def compare_repo_outputs(root: Path) -> tuple[bool, str]:
    r = run([PY, "scripts/compare_outputs.py", "outputs_validation/_pytest"], root)
    rep = (root / "outputs_validation" / "_pytest" / "COMPARISON.md")
    if r.returncode != 0 or not rep.exists():
        return False, r.stderr[-500:]
    text = rep.read_text(encoding="utf-8")
    # 파이프라인이 만들지 않는 것이 정상인 파일: 지오코딩 실행 기록(외부 API), 선택적 실제 LLM 측정 기록
    optional = ("geocod", "llm_grounding_check", "medical_boundary_exceptions")
    bad = [line for line in text.splitlines() if line.startswith("| DIFFERENT")
           or (line.startswith("| NOT_REGENERATED") and not any(o in line for o in optional))]
    return not bad, (text.splitlines()[2] if len(text.splitlines()) > 2 else "") + (f" | 차이: {bad[:3]}" if bad else "")


def package_gate() -> tuple[bool, str]:
    sys.path.insert(0, str(BASE / "scripts"))
    from make_submission_zip import main as make_zip
    z = make_zip([])
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        bad = [n for n in names if re.search(r"(^|/)(\.env$|\.env\.(?!example$)|\.git/|\.venv/|__pycache__/|"
                                             r"\.pytest_cache/|outputs_validation/|dist/|[^/]+\.zip$)", n)]
        if bad:
            return False, f"ZIP에 제외 대상 포함: {bad[:5]}"
        tmp = Path(tempfile.mkdtemp(prefix="mlzip_"))
        zf.extractall(tmp)
    root = tmp / "MobiLens-Seongnam"
    steps = []
    r = run([PY, "-c", "import src.config, src.accessibility, src.policy_scenarios, src.explanation; print('import ok')"], root)
    steps.append(("import", r.returncode == 0))
    ok_t, _, tail = pytest_results(root)
    steps.append(("pytest", ok_t))
    r = run([PY, "run_pipeline.py", "--demo"], root, {"MOBILENS_OUT_ROOT": ""})
    steps.append(("demo", r.returncode == 0))
    ok = all(s for _, s in steps)
    detail = ", ".join(f"{n}={'PASS' if s else 'FAIL'}" for n, s in steps) + f" | {z.name} ({len(names)} files) | {root}"
    if ok:
        shutil.rmtree(tmp, ignore_errors=True)
    elif not ok_t:
        detail += "\n" + tail
    return ok, detail


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    gates: dict[str, tuple[bool, str]] = {}
    req = ["README.md", "requirements.txt", "run_pipeline.py", "demo.py", "dashboard/app.py", ".env.example",
           "NOTICE.md", "docs/methodology.md", "docs/limitations.md", "data/raw/osm/osm_snapshot_meta.json",
           "data/SHA256SUMS",
           "data/processed/medical_facilities_geocoded.csv", "data/processed/care_facilities_geocoded.csv"]
    missing = [f for f in req if not (BASE / f).exists()]
    print(f"필수 파일: {'PASS' if not missing else 'FAIL ' + str(missing)}")
    print("pytest 실행 중 (raw부터 전체 파이프라인 포함)...", flush=True)
    ok_all, res, tail = pytest_results(BASE)
    for g in "ACDEFH":
        gates[g] = gate_from_tests(res, GATE_TESTS[g])
    okb, db = gate_from_tests(res, GATE_TESTS["B"])
    okc, dc = compare_repo_outputs(BASE)
    gates["B"] = (okb and okc and not missing, f"{db}; 저장소 outputs 비교: {dc}")
    gates["G"] = (ok_all, f"{sum(v == 'PASS' for v in res.values())}/{len(res)} passed")
    r = run([PY, "run_pipeline.py", "--demo", "--tamper"], BASE)
    gates["H"] = (gates["H"][0] and r.returncode == 0, gates["H"][1] + f"; CLI demo exit={r.returncode}")
    oki, issues = security_scan(BASE)
    gates["I"] = (oki, "이상 없음" if oki else "; ".join(issues[:8]))
    if "--package" in sys.argv:
        print("패키지 게이트: ZIP 생성·압축해제·재검증 중...", flush=True)
        gates["J"] = package_gate()
    else:
        gates["J"] = (False, "미실행 (--package 옵션 필요)")
    names = {"A": "DATA", "B": "REPRODUCIBILITY", "C": "ANALYTICS", "D": "OPTIMIZATION", "E": "VALIDATION",
             "F": "LLM", "G": "TEST", "H": "DEMO", "I": "SECURITY", "J": "PACKAGE"}
    print("\n===== ACCEPTANCE GATES =====")
    for g in "ABCDEFGHIJ":
        ok, d = gates[g]
        print(f"GATE {g} — {names[g]:16s} {'PASS' if ok else 'FAIL'}  {d}")
    if not ok_all:
        print(tail)
    if all(ok for ok, _ in gates.values()) and not missing:
        print("\nSUBMISSION READY\nMOBILENS FINAL RELEASE READY")
        return 0
    print("\nNOT READY")
    return 1


if __name__ == "__main__":
    sys.exit(main())
