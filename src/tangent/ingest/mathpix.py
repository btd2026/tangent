"""Optional Mathpix LaTeX OCR for equation / handwriting crops."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def mathpix_credentials() -> tuple[str, str] | None:
    app_id = os.environ.get("MATHPIX_APP_ID", "").strip()
    app_key = os.environ.get("MATHPIX_APP_KEY", "").strip()
    if app_id and app_key:
        return app_id, app_key
    return None


def latex_from_image(path: Path, *, timeout: float = 30.0) -> str | None:
    """Return LaTeX for an image crop, or None if Mathpix is unset / fails."""
    creds = mathpix_credentials()
    if creds is None:
        return None
    app_id, app_key = creds
    data = path.read_bytes()
    ext = path.suffix.lstrip(".").lower() or "png"
    if ext == "jpg":
        ext = "jpeg"
    b64 = base64.b64encode(data).decode("ascii")
    payload = {
        "src": f"data:image/{ext};base64,{b64}",
        "formats": ["latex_styled", "text"],
        "data_options": {"include_latex": True},
    }
    req = urllib.request.Request(
        "https://api.mathpix.com/v3/text",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "app_id": app_id,
            "app_key": app_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    latex = body.get("latex_styled") or body.get("latex") or body.get("text")
    if isinstance(latex, str) and latex.strip():
        return latex.strip()
    return None
