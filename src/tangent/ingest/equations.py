"""Equation / figure image classification from PDF placements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pymupdf

from .geometry import BBox, bbox_area, clamp_bbox


@dataclass
class PlacedImage:
    xref: int
    bbox: BBox
    width: int
    height: int
    has_smask: bool
    page_width: float
    page_height: float

    @property
    def placed_w(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def placed_h(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def page_frac(self) -> float:
        page_a = self.page_width * self.page_height
        return bbox_area(self.bbox) / page_a if page_a > 0 else 0.0


def _xref_has_smask(doc: pymupdf.Document, xref: int) -> bool:
    try:
        obj = doc.xref_object(xref)
    except Exception:
        return False
    return "/SMask" in obj


def collect_placed_images(page: pymupdf.Page, doc: pymupdf.Document) -> list[PlacedImage]:
    pw, ph = page.rect.width, page.rect.height
    out: list[PlacedImage] = []
    for info in page.get_image_info(xrefs=True):
        xref = int(info.get("xref") or 0)
        if xref <= 0:
            continue
        bbox = clamp_bbox(info["bbox"], pw, ph)
        if bbox_area(bbox) <= 1:
            continue
        out.append(
            PlacedImage(
                xref=xref,
                bbox=bbox,
                width=int(info.get("width") or 0),
                height=int(info.get("height") or 0),
                has_smask=_xref_has_smask(doc, xref),
                page_width=pw,
                page_height=ph,
            )
        )
    return out


def text_gap_y_bands(text_dict: dict[str, Any], page_height: float) -> list[tuple[float, float]]:
    """Return approximate horizontal text line y-ranges for gap filling."""
    bands: list[tuple[float, float]] = []
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            bbox = line.get("bbox")
            if not bbox:
                continue
            y0, y1 = float(bbox[1]), float(bbox[3])
            if y1 > y0:
                bands.append((y0, y1))
    bands.sort()
    return bands


def is_equation_candidate(
    img: PlacedImage,
    *,
    text_bands: list[tuple[float, float]],
    image_page: bool,
) -> bool:
    """Heuristic equation detector that does not assume a specific PDF producer."""
    if image_page:
        # On flattened pages, small/medium strips can still be equation crops,
        # but full-bleed tiles are not.
        if img.page_frac > 0.2:
            return False

    h_frac = img.placed_h / img.page_height if img.page_height else 0
    w_frac = img.placed_w / img.page_width if img.page_width else 0
    aspect = img.placed_w / img.placed_h if img.placed_h > 1e-6 else 0

    # Soft-masked short/wide placements are classic embedded math screenshots.
    if img.has_smask and h_frac < 0.18 and aspect >= 1.2 and img.page_frac < 0.25:
        return True

    # Inline strip in a text band (fills the "Let ____ be" holes).
    if h_frac < 0.12 and aspect >= 1.5 and img.page_frac < 0.15:
        cy = 0.5 * (img.bbox[1] + img.bbox[3])
        for y0, y1 in text_bands:
            if y0 - 4 <= cy <= y1 + 4:
                return True

    # Tall-ish but still formula-sized (stacked fractions / column vectors)
    if img.has_smask and h_frac < 0.35 and w_frac < 0.55 and img.page_frac < 0.12:
        return True

    return False


def is_figure_candidate(img: PlacedImage, *, equation: bool, image_page: bool) -> bool:
    if equation:
        return False
    if image_page and img.page_frac > 0.15:
        # Full-page / tile imagery — represent once as a page figure, not every tile.
        return img.page_frac > 0.45 or (
            img.bbox[0] < 2 and img.bbox[1] < 2 and img.page_frac > 0.2
        )
    # Skip tiny icons / bullets
    if img.page_frac < 0.01 and img.placed_h < 24 and img.placed_w < 24:
        return False
    if img.page_frac < 0.008:
        return False
    return True


def dedupe_image_regions(
    items: list[tuple[PlacedImage, str]],
    *,
    iou_thresh: float = 0.7,
) -> list[tuple[PlacedImage, str]]:
    """Drop near-duplicate placements (same visual, overlapping boxes)."""
    from .geometry import bbox_iou

    kept: list[tuple[PlacedImage, str]] = []
    for img, kind in sorted(items, key=lambda t: -bbox_area(t[0].bbox)):
        if any(bbox_iou(img.bbox, k[0].bbox) >= iou_thresh and k[1] == kind for k in kept):
            continue
        kept.append((img, kind))
    return kept
