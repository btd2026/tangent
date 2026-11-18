"""Fixture PDF discovery. Lecture PDFs are not committed; resolve from env/store."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

STORE_INTERNAL = Path("/cursor/stores/self/internal")

FIXTURE_FILES = {
    "cs_image_processing": "lecture_04_preliminaries-image_processing.pdf",
    "cs_transformations": "lecture_02_preliminaries-transformations.pdf",
    "bio_vaccinology": "lecture_01_ciaramella.pdf",
}


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("TANGENT_FIXTURE_DIR", "").strip()
    if env:
        dirs.append(Path(env).expanduser())
    dirs.append(STORE_INTERNAL)
    # Optional local (gitignored) copies for developers
    dirs.append(Path(__file__).resolve().parent / "fixtures")
    return dirs


def resolve_fixture(name: str) -> Path | None:
    filename = FIXTURE_FILES[name]
    for d in _candidate_dirs():
        path = d / filename
        if path.is_file():
            return path
    return None


@pytest.fixture(scope="session")
def fixture_pdfs() -> dict[str, Path]:
    found = {k: resolve_fixture(k) for k in FIXTURE_FILES}
    missing = [k for k, v in found.items() if v is None]
    if missing:
        pytest.skip(
            "Missing fixture PDFs: "
            + ", ".join(FIXTURE_FILES[m] for m in missing)
            + ". Set TANGENT_FIXTURE_DIR or place files under /cursor/stores/self/internal/"
        )
    assert all(v is not None for v in found.values())
    return {k: v for k, v in found.items() if v is not None}
