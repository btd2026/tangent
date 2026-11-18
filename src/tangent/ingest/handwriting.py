"""Separate handwriting / ink from printed slide content."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .geometry import BBox, clamp_bbox


@dataclass
class InkCluster:
    bbox_px: BBox  # pixel coords in the rendered image
    color: str  # "red" | "blue" | "black"
    pixel_count: int


def _connected_components(mask: np.ndarray, *, max_components: int = 400) -> list[tuple[BBox, int]]:
    """4-connected component labeling for boolean masks (expects modest resolution)."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[tuple[BBox, int]] = []
    ys, xs = np.where(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        min_x = max_x = x
        min_y = max_y = y
        count = 0
        while stack:
            cy, cx = stack.pop()
            count += 1
            if cx < min_x:
                min_x = cx
            if cx > max_x:
                max_x = cx
            if cy < min_y:
                min_y = cy
            if cy > max_y:
                max_y = cy
            for ny, nx in (
                (cy - 1, cx),
                (cy + 1, cx),
                (cy, cx - 1),
                (cy, cx + 1),
            ):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        components.append(((min_x, min_y, max_x + 1, max_y + 1), count))
        if len(components) >= max_components:
            break
    return components


def _merge_nearby(
    clusters: list[InkCluster],
    *,
    gap: int,
    same_color: bool = True,
) -> list[InkCluster]:
    if not clusters:
        return []
    changed = True
    current = list(clusters)
    while changed:
        changed = False
        merged: list[InkCluster] = []
        used = [False] * len(current)
        for i, a in enumerate(current):
            if used[i]:
                continue
            ax0, ay0, ax1, ay1 = a.bbox_px
            color = a.color
            count = a.pixel_count
            for j in range(i + 1, len(current)):
                if used[j]:
                    continue
                b = current[j]
                if same_color and b.color != color:
                    continue
                bx0, by0, bx1, by1 = b.bbox_px
                if (
                    ax0 - gap <= bx1
                    and bx0 - gap <= ax1
                    and ay0 - gap <= by1
                    and by0 - gap <= ay1
                ):
                    ax0, ay0 = min(ax0, bx0), min(ay0, by0)
                    ax1, ay1 = max(ax1, bx1), max(ay1, by1)
                    count += b.pixel_count
                    used[j] = True
                    changed = True
            merged.append(InkCluster((ax0, ay0, ax1, ay1), color, count))
            used[i] = True
        current = merged
    return current


def color_ink_masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Detect saturated red / blue pen ink on light slide backgrounds."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    # Strong chroma (pen) rather than mild plot tints
    red = (r > 140) & ((r - g) > 50) & ((r - b) > 50) & (g < 120) & (b < 120)
    blue = (b > 140) & ((b - r) > 40) & ((b - g) > 25) & (r < 120) & (g < 160)
    # Prefer strokes drawn on light paper / slide chrome
    light = (r > 200) & (g > 200) & (b > 200)
    # Dilate light mask roughly via shifts to require nearby white
    near_light = light.copy()
    near_light[1:, :] |= light[:-1, :]
    near_light[:-1, :] |= light[1:, :]
    near_light[:, 1:] |= light[:, :-1]
    near_light[:, :-1] |= light[:, 1:]
    return {"red": red & near_light, "blue": blue & near_light}


def black_ink_mask(rgb: np.ndarray, *, max_value: int = 70) -> np.ndarray:
    """Dark strokes on light backgrounds (board / flattened handwriting pages)."""
    return np.all(rgb < max_value, axis=2)


def find_ink_clusters(
    rgb: np.ndarray,
    *,
    image_page: bool,
    min_pixels: int = 40,
    min_area: int = 80,
    merge_gap: int = 18,
    work_max_side: int = 720,
) -> list[InkCluster]:
    """Return ink clusters in full-resolution pixel space for a rendered page."""
    h, w = rgb.shape[:2]
    scale = 1.0
    work = rgb
    if max(h, w) > work_max_side:
        scale = work_max_side / float(max(h, w))
        small = Image.fromarray(rgb).resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.BILINEAR,
        )
        work = np.asarray(small)

    inv = 1.0 / scale if scale > 0 else 1.0
    # Scale thresholds into work space
    min_px = max(8, int(min_pixels * (scale**2)))
    min_ar = max(16, int(min_area * (scale**2)))
    gap = max(4, int(merge_gap * scale))

    clusters: list[InkCluster] = []
    color_masks = color_ink_masks(work)
    for color, mask in color_masks.items():
        if not mask.any():
            continue
        for bbox, count in _connected_components(mask):
            if count < min_px:
                continue
            x0, y0, x1, y1 = bbox
            area = (x1 - x0) * (y1 - y0)
            if area < min_ar:
                continue
            if area > 0.45 * work.shape[0] * work.shape[1]:
                continue
            fill = count / area if area else 0
            # Reject solid color blocks (diagram tiles); keep stroke-like ink
            if fill > 0.55:
                continue
            full = (
                int(x0 * inv),
                int(y0 * inv),
                int(x1 * inv),
                int(y1 * inv),
            )
            clusters.append(InkCluster(full, color, int(count / max(scale**2, 1e-6))))

    # Black-ink path is for flattened / board pages only. Running it on
    # ordinary digital slides false-positives on photos and dark UI chrome.
    if image_page:
        mask = black_ink_mask(work)
        frac = float(mask.mean()) if mask.size else 0.0
        if 0.002 < frac < 0.25:
            for bbox, count in _connected_components(mask):
                if count < max(min_px, 40):
                    continue
                x0, y0, x1, y1 = bbox
                area = (x1 - x0) * (y1 - y0)
                if area < min_ar * 2:
                    continue
                fill = count / area if area else 0
                if area > 0.35 * work.shape[0] * work.shape[1]:
                    continue
                if fill > 0.85 and area > 2500:
                    continue
                full = (
                    int(x0 * inv),
                    int(y0 * inv),
                    int(x1 * inv),
                    int(y1 * inv),
                )
                clusters.append(
                    InkCluster(full, "black", int(count / max(scale**2, 1e-6)))
                )

    return _merge_nearby(clusters, gap=max(8, int(merge_gap)), same_color=True)


def px_bbox_to_pdf(
    bbox_px: BBox,
    *,
    page_width: float,
    page_height: float,
    render_width: int,
    render_height: int,
) -> BBox:
    sx = page_width / max(render_width, 1)
    sy = page_height / max(render_height, 1)
    return clamp_bbox(
        (bbox_px[0] * sx, bbox_px[1] * sy, bbox_px[2] * sx, bbox_px[3] * sy),
        page_width,
        page_height,
    )


def render_to_numpy(pil_image: Image.Image) -> np.ndarray:
    return np.asarray(pil_image.convert("RGB"))
