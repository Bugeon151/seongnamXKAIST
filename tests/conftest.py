"""pytest 공통 설정.

통합 테스트는 저장소 outputs/를 건드리지 않도록 outputs_validation/_pytest 에 파이프라인을 새로 실행한다
(세션당 1회). MOBILENS_TEST_REUSE=1 이고 이전 실행이 성공했다면 재사용한다(개발 중 반복 실행용).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
TEST_ROOT = BASE / "outputs_validation" / "_pytest"
os.environ["MOBILENS_OUT_ROOT"] = str(TEST_ROOT)
os.environ["OLLAMA_MODEL"] = ""  # 테스트는 LLM 없이 결정적으로 (.env보다 우선, LLM 경로는 mock으로 검증)
sys.path.insert(0, str(BASE))


@pytest.fixture(scope="session")
def pipeline_run() -> Path:
    status = TEST_ROOT / "outputs" / "logs" / "pipeline_status.csv"
    reuse = os.environ.get("MOBILENS_TEST_REUSE") == "1" and (TEST_ROOT / "outputs" / "run_metadata.json").exists()
    if not reuse:
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, str(BASE / "run_pipeline.py"), "--out", str(TEST_ROOT)], cwd=BASE,
                           env=env, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            pytest.fail(f"파이프라인 실패:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
    assert status.exists()
    return TEST_ROOT


def load_json(root: Path, rel: str):
    return json.loads((root / "outputs" / rel).read_text(encoding="utf-8"))
