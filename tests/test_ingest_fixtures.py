"""End-to-end ingest checks on CS + biology lecture PDFs (not committed)."""

from __future__ import annotations

from pathlib import Path

from tangent.ingest.pipeline import IngestConfig, ingest_pdf


def _counts(index: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for page in index["pages"]:
        for r in page["regions"]:
            out[r["type"]] = out.get(r["type"], 0) + 1
    return out


def _run(pdf: Path, tmp_path: Path, name: str) -> dict:
    out = tmp_path / name
    result = ingest_pdf(
        pdf,
        out,
        IngestConfig(call_mathpix=False, render_scale=1.5),
    )
    assert result.index_path.is_file()
    assert result.index["page_count"] == len(result.index["pages"])
    assert result.index["page_count"] >= 1
    for page in result.index["pages"]:
        assert "regions" in page
        assert page["mode"] in {"digital", "image"}
        for r in page["regions"]:
            assert r["type"] in {"text", "equation", "handwriting", "figure"}
            assert len(r["bbox"]) == 4
            assert len(r["bbox_norm"]) == 4
            assert all(0 <= c <= 1.01 for c in r["bbox_norm"])
            if r["type"] == "text":
                assert r["text"]
            if r["type"] == "equation":
                assert r["crop_path"]
                assert (out / r["crop_path"]).is_file()
                assert r["latex"] is None  # mathpix disabled in tests
            if r["type"] == "handwriting":
                assert r["ink_color"] in {"red", "blue", "black"}
                assert r["crop_path"]
            if r["type"] == "figure":
                assert r["crop_path"]
                assert (out / r["crop_path"]).is_file()
    return result.index


def test_ingest_cs_image_processing(fixture_pdfs, tmp_path):
    index = _run(fixture_pdfs["cs_image_processing"], tmp_path, "cs_ip")
    counts = _counts(index)
    assert index["page_count"] == 26
    assert counts.get("text", 0) >= 20
    # Soft-mask / gap equations should surface
    assert counts.get("equation", 0) >= 5
    assert counts.get("figure", 0) >= 5
    # Clean Google export: expect little/no handwriting
    assert counts.get("handwriting", 0) <= 15


def test_ingest_cs_transformations(fixture_pdfs, tmp_path):
    index = _run(fixture_pdfs["cs_transformations"], tmp_path, "cs_xf")
    counts = _counts(index)
    assert index["page_count"] == 25
    assert counts.get("text", 0) >= 15
    assert counts.get("equation", 0) >= 3
    # Annotated / ink-heavy deck
    assert counts.get("handwriting", 0) >= 5
    image_pages = [p for p in index["pages"] if p["mode"] == "image"]
    assert len(image_pages) >= 1
    # Flattened pages should still yield regions (figure and/or handwriting)
    for p in image_pages:
        assert len(p["regions"]) >= 1


def test_ingest_bio_vaccinology(fixture_pdfs, tmp_path):
    """Biology lecture (Ciaramella / mRNA vaccines) — text + figures, optional eqs."""
    index = _run(fixture_pdfs["bio_vaccinology"], tmp_path, "bio")
    counts = _counts(index)
    assert index["page_count"] == 35
    assert counts.get("text", 0) >= 30
    assert counts.get("figure", 0) >= 1
    # Must produce a usable region index even if equations/handwriting are sparse
    assert sum(counts.values()) >= 30
    # Spot-check: title/intro pages have text regions
    p1 = index["pages"][0]
    assert p1["mode"] == "digital"
    assert any(r["type"] == "text" for r in p1["regions"])


def test_cli_smoke(fixture_pdfs, tmp_path):
    from tangent.ingest.cli import main

    out = tmp_path / "cli_out"
    rc = main(
        [
            str(fixture_pdfs["bio_vaccinology"]),
            "-o",
            str(out),
            "--no-mathpix",
            "--render-scale",
            "1.25",
        ]
    )
    assert rc == 0
    assert (out / "regions.json").is_file()
