"""Geometry helpers for page regions."""

from __future__ import annotations

from typing import Iterable, Sequence


BBox = tuple[float, float, float, float]


def clamp_bbox(bbox: Sequence[float], width: float, height: float) -> BBox:
    x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    x0 = max(0.0, min(x0, width))
    y0 = max(0.0, min(y0, height))
    x1 = max(0.0, min(x1, width))
    y1 = max(0.0, min(y1, height))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def normalize_bbox(bbox: Sequence[float], width: float, height: float) -> BBox:
    x0, y0, x1, y1 = clamp_bbox(bbox, width, height)
    if width <= 0 or height <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return x0 / width, y0 / height, x1 / width, y1 / height


def bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0, float(bbox[3]) - float(bbox[1])
    )


def bbox_intersection(a: Sequence[float], b: Sequence[float]) -> BBox | None:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = bbox_intersection(a, b)
    if inter is None:
        return 0.0
    inter_a = bbox_area(inter)
    union = bbox_area(a) + bbox_area(b) - inter_a
    return inter_a / union if union > 0 else 0.0


def expand_bbox(bbox: Sequence[float], pad: float, width: float, height: float) -> BBox:
    return clamp_bbox(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        width,
        height,
    )


def merge_bboxes(boxes: Iterable[Sequence[float]]) -> BBox | None:
    boxes = list(boxes)
    if not boxes:
        return None
    return (
        min(float(b[0]) for b in boxes),
        min(float(b[1]) for b in boxes),
        max(float(b[2]) for b in boxes),
        max(float(b[3]) for b in boxes),
    )


def overlaps_any(bbox: Sequence[float], others: Iterable[Sequence[float]], iou: float = 0.3) -> bool:
    return any(bbox_iou(bbox, o) >= iou for o in others)
