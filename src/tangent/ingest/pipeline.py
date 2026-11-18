"""Ingest any slide-deck PDF into a JSON region index."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image

from .equations import (
    collect_placed_images,
    dedupe_image_regions,
    is_equation_candidate,
    is_figure_candidate,
    text_gap_y_bands,
)
from .geometry import (
    bbox_area,
    bbox_intersection,
    clamp_bbox,
    expand_bbox,
    normalize_bbox,
)
from .handwriting import find_ink_clusters, px_bbox_to_pdf, render_to_numpy
from .mathpix import latex_from_image, mathpix_credentials


FOOTER_RE = re.compile(
    r"(alex\.wong@|@yale\.edu|^\d+$|page\s*\d+)",
    re.IGNORECASE,
)


@dataclass
class IngestConfig:
    render_scale: float = 2.0
    call_mathpix: bool = True
    max_mathpix: int | None = None  # None = all equation crops
    save_page_previews: bool = False


@dataclass
class IngestResult:
    index: dict[str, Any]
    output_dir: Path
    index_path: Path
    mathpix_enabled: bool
    warnings: list[str] = field(default_factory=list)


def _is_image_page(char_count: int, images: list, page_area: float) -> bool:
    if char_count >= 40:
        return False
    if not images:
        return char_count == 0
    covered = sum(bbox_area(im.bbox) for im in images)
    # Tiled full-bleed pages often over-count area; still treat as image page.
    return covered >= 0.35 * page_area or (char_count == 0 and len(images) >= 1)


def _extract_text_regions(page: pymupdf.Page, page_idx: int) -> list[dict[str, Any]]:
    pw, ph = page.rect.width, page.rect.height
    regions: list[dict[str, Any]] = []
    blocks = page.get_text("dict").get("blocks", [])
    for bi, block in enumerate(blocks):
        if block.get("type") != 0:
            continue
        parts: list[str] = []
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", []))
            if line_text.strip():
                parts.append(line_text.strip())
        text = "\n".join(parts).strip()
        if not text:
            continue
        # Drop trivial footers / lone page numbers when tiny
        bbox = clamp_bbox(block["bbox"], pw, ph)
        if bbox_area(bbox) <= 1:
            continue
        if len(text) <= 3 and FOOTER_RE.search(text):
            continue
        regions.append(
            {
                "id": f"p{page_idx:03d}_text_{bi:03d}",
                "type": "text",
                "bbox": [round(c, 2) for c in bbox],
                "bbox_norm": [round(c, 5) for c in normalize_bbox(bbox, pw, ph)],
                "text": text,
                "latex": None,
                "crop_path": None,
                "ink_color": None,
                "meta": {"source": "pdf_text"},
            }
        )
    return regions


def _render_page(page: pymupdf.Page, scale: float) -> tuple[Image.Image, int, int]:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img, pix.width, pix.height


def _save_bbox_crop(
    page: pymupdf.Page,
    bbox: tuple[float, float, float, float],
    path: Path,
    *,
    scale: float = 3.0,
    pad: float = 2.0,
) -> None:
    pw, ph = page.rect.width, page.rect.height
    clip = pymupdf.Rect(expand_bbox(bbox, pad, pw, ph))
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(path))


def _save_smask_composite_crop(
    doc: pymupdf.Document,
    page: pymupdf.Page,
    xref: int,
    bbox: tuple[float, float, float, float],
    path: Path,
) -> None:
    """Prefer page clip render so soft-mask equations appear correctly."""
    _save_bbox_crop(page, bbox, path, scale=4.0, pad=1.0)


def ingest_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    config: IngestConfig | None = None,
) -> IngestResult:
    cfg = config or IngestConfig()
    pdf_path = Path(pdf_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    warnings: list[str] = []
    mathpix_on = bool(cfg.call_mathpix and mathpix_credentials())
    if cfg.call_mathpix and not mathpix_on:
        warnings.append(
            "MATHPIX_APP_ID/MATHPIX_APP_KEY not set; equation latex fields left null"
        )

    pages_out: list[dict[str, Any]] = []
    mathpix_calls = 0

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        pw, ph = float(page.rect.width), float(page.rect.height)
        page_num = page_idx + 1
        text = page.get_text("text")
        text_dict = page.get_text("dict")
        placed = collect_placed_images(page, doc)
        image_page = _is_image_page(len(text.strip()), placed, pw * ph)
        bands = text_gap_y_bands(text_dict, ph)

        regions: list[dict[str, Any]] = []
        if not image_page:
            regions.extend(_extract_text_regions(page, page_num))

        classified: list[tuple[Any, str]] = []
        for img in placed:
            eq = is_equation_candidate(img, text_bands=bands, image_page=image_page)
            if eq:
                classified.append((img, "equation"))
            elif is_figure_candidate(img, equation=False, image_page=image_page):
                classified.append((img, "figure"))
        classified = dedupe_image_regions(classified)

        # On heavily tiled image pages, keep a single page-level figure if none survived.
        if image_page and not any(k == "figure" for _, k in classified):
            # Union of all image boxes as one figure region
            if placed:
                from .geometry import merge_bboxes

                ub = merge_bboxes(im.bbox for im in placed)
                if ub:
                    crop_rel = f"crops/p{page_num:03d}_figure_page.png"
                    _save_bbox_crop(page, ub, output_dir / crop_rel, scale=2.0)
                    regions.append(
                        {
                            "id": f"p{page_num:03d}_figure_page",
                            "type": "figure",
                            "bbox": [round(c, 2) for c in ub],
                            "bbox_norm": [round(c, 5) for c in normalize_bbox(ub, pw, ph)],
                            "text": None,
                            "latex": None,
                            "crop_path": crop_rel,
                            "ink_color": None,
                            "meta": {"source": "image_page", "tile_count": len(placed)},
                        }
                    )

        eq_i = fig_i = 0
        for img, kind in classified:
            if kind == "equation":
                eq_i += 1
                crop_rel = f"crops/p{page_num:03d}_eq_{eq_i:03d}.png"
                _save_smask_composite_crop(doc, page, img.xref, img.bbox, output_dir / crop_rel)
                latex = None
                if mathpix_on and (
                    cfg.max_mathpix is None or mathpix_calls < cfg.max_mathpix
                ):
                    latex = latex_from_image(output_dir / crop_rel)
                    mathpix_calls += 1
                regions.append(
                    {
                        "id": f"p{page_num:03d}_eq_{eq_i:03d}",
                        "type": "equation",
                        "bbox": [round(c, 2) for c in img.bbox],
                        "bbox_norm": [
                            round(c, 5) for c in normalize_bbox(img.bbox, pw, ph)
                        ],
                        "text": None,
                        "latex": latex,
                        "crop_path": crop_rel,
                        "ink_color": None,
                        "meta": {
                            "xref": img.xref,
                            "has_smask": img.has_smask,
                            "source": "image_placement",
                        },
                    }
                )
            else:
                fig_i += 1
                crop_rel = f"crops/p{page_num:03d}_fig_{fig_i:03d}.png"
                _save_bbox_crop(page, img.bbox, output_dir / crop_rel, scale=2.0)
                regions.append(
                    {
                        "id": f"p{page_num:03d}_fig_{fig_i:03d}",
                        "type": "figure",
                        "bbox": [round(c, 2) for c in img.bbox],
                        "bbox_norm": [
                            round(c, 5) for c in normalize_bbox(img.bbox, pw, ph)
                        ],
                        "text": None,
                        "latex": None,
                        "crop_path": crop_rel,
                        "ink_color": None,
                        "meta": {
                            "xref": img.xref,
                            "has_smask": img.has_smask,
                            "source": "image_placement",
                        },
                    }
                )

        # Handwriting / ink
        render, rw, rh = _render_page(page, cfg.render_scale)
        rgb = render_to_numpy(render)
        ink_clusters = find_ink_clusters(rgb, image_page=image_page)
        figure_boxes = [r["bbox"] for r in regions if r["type"] == "figure"]
        hi = 0
        for cluster in ink_clusters:
            pdf_bbox = px_bbox_to_pdf(
                cluster.bbox_px,
                page_width=pw,
                page_height=ph,
                render_width=rw,
                render_height=rh,
            )
            if bbox_area(pdf_bbox) < 8:
                continue
            # Suppress plot/photo colors that sit inside figure placements
            ink_a = bbox_area(pdf_bbox)
            inside_figure = False
            for fb in figure_boxes:
                inter = bbox_intersection(pdf_bbox, fb)
                if inter and bbox_area(inter) >= 0.6 * ink_a:
                    inside_figure = True
                    break
            if inside_figure and not image_page:
                continue
            hi += 1
            crop_rel = f"crops/p{page_num:03d}_ink_{hi:03d}_{cluster.color}.png"
            _save_bbox_crop(page, pdf_bbox, output_dir / crop_rel, scale=3.0, pad=3.0)
            latex = None
            # Optional: Mathpix on ink only when credentials present and still under budget
            if (
                mathpix_on
                and cluster.color in {"red", "blue", "black"}
                and (cfg.max_mathpix is None or mathpix_calls < cfg.max_mathpix)
                and image_page
            ):
                latex = latex_from_image(output_dir / crop_rel)
                mathpix_calls += 1
            regions.append(
                {
                    "id": f"p{page_num:03d}_ink_{hi:03d}",
                    "type": "handwriting",
                    "bbox": [round(c, 2) for c in pdf_bbox],
                    "bbox_norm": [
                        round(c, 5) for c in normalize_bbox(pdf_bbox, pw, ph)
                    ],
                    "text": None,
                    "latex": latex,
                    "crop_path": crop_rel,
                    "ink_color": cluster.color,
                    "meta": {
                        "pixel_count": cluster.pixel_count,
                        "source": "color_ink" if cluster.color != "black" else "dark_ink",
                    },
                }
            )

        if cfg.save_page_previews:
            preview = output_dir / "previews" / f"p{page_num:03d}.png"
            preview.parent.mkdir(parents=True, exist_ok=True)
            render.save(preview)

        pages_out.append(
            {
                "page": page_num,
                "width": round(pw, 2),
                "height": round(ph, 2),
                "mode": "image" if image_page else "digital",
                "char_count": len(text),
                "regions": regions,
            }
        )

    index = {
        "schema_version": 1,
        "source": str(pdf_path),
        "source_name": pdf_path.name,
        "page_count": doc.page_count,
        "metadata": {
            "title": doc.metadata.get("title") or "",
            "author": doc.metadata.get("author") or "",
            "creator": doc.metadata.get("creator") or "",
            "producer": doc.metadata.get("producer") or "",
            "format": doc.metadata.get("format") or "",
        },
        "mathpix_enabled": mathpix_on,
        "pages": pages_out,
    }
    index_path = output_dir / "regions.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    doc.close()
    return IngestResult(
        index=index,
        output_dir=output_dir,
        index_path=index_path,
        mathpix_enabled=mathpix_on,
        warnings=warnings,
    )
