from tangent.ingest.geometry import bbox_iou, clamp_bbox, normalize_bbox


def test_clamp_and_normalize():
    assert clamp_bbox((-10, -10, 50, 60), 100, 200) == (0, 0, 50, 60)
    n = normalize_bbox((0, 0, 50, 100), 100, 200)
    assert n == (0.0, 0.0, 0.5, 0.5)


def test_iou():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
