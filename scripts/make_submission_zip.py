"""제출용 코드 원본 ZIP 생성.

사용: python scripts/make_submission_zip.py [출력경로]
기본 출력: dist/AI경진대회_MobiLens_MobiLens-Seongnam_코드원본.zip
- 파일명은 UTF-8 플래그로 저장(한글 경로 깨짐 방지), ZIP 최상위 폴더 MobiLens-Seongnam/
- 저장소 폴더를 직접 순회하며 아래 제외 규칙을 적용 (git이 없는 환경에서도 동일 결과)
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ZIP_NAME = "AI경진대회_MobiLens_MobiLens-Seongnam_코드원본.zip"
ROOT_NAME = "MobiLens-Seongnam"
EXCLUDE_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                ".idea", ".vscode", ".claude", "outputs_validation", "dist", "build", "cache", "geocode_cache",
                "interim", ".ipynb_checkpoints", ".streamlit", ".tox"}
EXCLUDE_FILES = {".env", ".DS_Store", "Thumbs.db", "desktop.ini"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".log", ".zip", ".tmp", ".swp", ".bak"}


def included(rel: Path) -> bool:
    if any(part in EXCLUDE_DIRS or part.startswith(".venv") for part in rel.parts[:-1]):
        return False
    if rel.name.startswith(".env") and rel.name != ".env.example":  # .env, .env.local 등 실제 설정은 제외
        return False
    return rel.name not in EXCLUDE_FILES and rel.suffix.lower() not in EXCLUDE_SUFFIX


def collect() -> list[Path]:
    return sorted(p.relative_to(BASE) for p in BASE.rglob("*") if p.is_file() and included(p.relative_to(BASE)))


def main(argv: list[str] | None = None) -> Path:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]).resolve() if argv else BASE / "dist" / ZIP_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    files = collect()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            z.write(BASE / rel, f"{ROOT_NAME}/{rel.as_posix()}")
    size = out.stat().st_size / 1e6
    print(f"{out} ({len(files)} files, {size:.1f} MB)")
    return out


if __name__ == "__main__":
    main()
