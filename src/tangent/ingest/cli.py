"""CLI: tangent-ingest <pdf> -o <outdir>"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import IngestConfig, ingest_pdf


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tangent-ingest",
        description=(
            "Ingest a slide-deck PDF into a per-page JSON region index "
            "(text, equations, handwriting, figures)."
        ),
    )
    p.add_argument("pdf", type=Path, help="Path to a slide-deck PDF")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory for regions.json and crops/",
    )
    p.add_argument(
        "--render-scale",
        type=float,
        default=2.0,
        help="Rasterization scale for ink detection (default: 2.0)",
    )
    p.add_argument(
        "--no-mathpix",
        action="store_true",
        help="Do not call Mathpix even if MATHPIX_APP_ID/KEY are set",
    )
    p.add_argument(
        "--max-mathpix",
        type=int,
        default=None,
        help="Optional cap on Mathpix API calls this run",
    )
    p.add_argument(
        "--save-page-previews",
        action="store_true",
        help="Also write previews/pXXX.png for debugging",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = IngestConfig(
        render_scale=args.render_scale,
        call_mathpix=not args.no_mathpix,
        max_mathpix=args.max_mathpix,
        save_page_previews=args.save_page_previews,
    )
    try:
        result = ingest_pdf(args.pdf, args.output, cfg)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 — CLI boundary
        print(f"error: ingest failed: {e}", file=sys.stderr)
        return 1

    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)

    pages = result.index["page_count"]
    n_regions = sum(len(p["regions"]) for p in result.index["pages"])
    by_type: dict[str, int] = {}
    for page in result.index["pages"]:
        for r in page["regions"]:
            by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    print(f"Wrote {result.index_path}")
    print(f"Pages: {pages}  Regions: {n_regions}  {by_type}")
    print(f"Mathpix: {'enabled' if result.mathpix_enabled else 'disabled'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
