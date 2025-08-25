import io
import json
import os
import subprocess
import sys
from pathlib import Path


def _make_sample_pdf(tmp: Path) -> Path:
    # Generate a simple PDF using PyMuPDF to avoid heavy deps
    import fitz  # PyMuPDF

    path = tmp / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Sample Heading", fontsize=24)
    page.insert_text((72, 120), "This is a test block of text.", fontsize=12)
    doc.save(path)
    doc.close()
    return path


def test_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    pdf = _make_sample_pdf(tmp_path)
    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    cmd = [sys.executable, str(root / "pdf2mm.py"), "--pdf", str(pdf), "--outdir", str(outdir), "--format", "jsonl", "--caption-provider", "none", "--embed-provider", "none"]
    proc = subprocess.run(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.returncode == 0, proc.stderr

    jsonl = outdir / f"{pdf.stem}.jsonl"
    assert jsonl.exists(), "data.jsonl not found"
    lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    # Validate JSON lines
    first = json.loads(lines[0])
    assert "doc_id" in first and "page" in first and "unit" in first

    # If images present, ensure they exist on disk
    for line in lines:
        row = json.loads(line)
        if row.get("unit") == "image" and row.get("image_ref"):
            img_path = outdir / row["image_ref"]
            assert img_path.exists(), f"Missing image file {img_path}"

