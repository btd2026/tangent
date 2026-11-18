# tangent

A live note taking app for STEM lectures.

## Slide-deck ingest CLI

`tangent-ingest` turns **any** slide-deck PDF into a per-page JSON region index:

- **text** — selectable PDF text blocks (`bbox` + `text`)
- **equation** — candidate math regions (`bbox` + crop; `latex` via Mathpix when configured)
- **handwriting** — color ink (red/blue) clusters, plus dark-ink clusters on flattened image-only pages
- **figure** — non-equation image placements / full-bleed image pages

It does **not** assume Google Slides or iOS exports. The same path handles digital text PDFs, soft-mask equation images, and flattened screenshot pages.

### Install

```bash
python -m pip install -e ".[dev]"
```

### Run

```bash
tangent-ingest /path/to/deck.pdf -o /tmp/tangent-out
```

Output:

- `/tmp/tangent-out/regions.json` — region index
- `/tmp/tangent-out/crops/` — equation / figure / ink crops

Optional Mathpix (leave `latex` null if unset):

```bash
export MATHPIX_APP_ID=...
export MATHPIX_APP_KEY=...
tangent-ingest deck.pdf -o /tmp/tangent-out
```

Useful flags:

- `--no-mathpix` — never call the API
- `--max-mathpix N` — cap API calls
- `--render-scale 2.0` — raster scale for ink detection
- `--save-page-previews` — also write `previews/pXXX.png`

Or as a module:

```bash
python -m tangent.ingest.cli deck.pdf -o /tmp/tangent-out --no-mathpix
```

### Sample decks (not in git)

Place fixture PDFs somewhere local (or set `TANGENT_FIXTURE_DIR`) and run:

```bash
# CS — image processing (soft-mask equations, figures)
tangent-ingest "$TANGENT_FIXTURE_DIR/lecture_04_preliminaries-image_processing.pdf" -o /tmp/t-cs-ip --no-mathpix

# CS — transformations (handwriting + equation A/B pages)
tangent-ingest "$TANGENT_FIXTURE_DIR/lecture_02_preliminaries-transformations.pdf" -o /tmp/t-cs-xf --no-mathpix

# Biology — vaccinology / mRNA (text-heavy + figures)
tangent-ingest "$TANGENT_FIXTURE_DIR/lecture_01_ciaramella.pdf" -o /tmp/t-bio --no-mathpix
```

### Tests

```bash
export TANGENT_FIXTURE_DIR=/path/to/folder/with/the/three/pdfs
pytest -q
```

Fixtures expected (filenames only; **do not commit PDFs or Mathpix secrets**):

| Key | File |
|---|---|
| CS image processing | `lecture_04_preliminaries-image_processing.pdf` |
| CS transformations | `lecture_02_preliminaries-transformations.pdf` |
| Biology vaccinology | `lecture_01_ciaramella.pdf` |
