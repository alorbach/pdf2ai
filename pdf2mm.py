#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import fitz  # PyMuPDF
from PIL import Image, ImageFilter
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm


# ---------- Config Models ----------


class OpenAIConfig(BaseModel):
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None


class RuntimeConfig(BaseModel):
    language: str = "eng"
    caption_provider: str = "none"  # openai|blip|none
    caption_model: str = "gpt-4o-mini"
    embed_provider: str = "none"  # openai|hf|none
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_pages: int = 0
    min_caption_len: int = 6
    ocr: str = "auto"  # auto|yes|no
    openai: OpenAIConfig = OpenAIConfig()


# ---------- Utility ----------


def setup_logger(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    return float(np.percentile(arr, pct))


def ensure_out_dirs(outdir: Path) -> Dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    images_dir = outdir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    embeds_dir = outdir / "embeddings"
    embeds_dir.mkdir(parents=True, exist_ok=True)
    return {"images": images_dir, "embeddings": embeds_dir}


def guess_bold(font_name: str) -> bool:
    name = (font_name or "").lower()
    return "bold" in name or name.endswith("-bd") or name.endswith("-bold")


def compute_edge_density(pil_img: Image.Image) -> float:
    # Simple heuristic: higher edge density indicates likely text presence
    gray = pil_img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges, dtype=np.uint8)
    threshold = 20
    high_edges = (arr > threshold).sum()
    total = arr.size
    return float(high_edges) / float(total) if total else 0.0


def should_ocr(pil_img: Image.Image, mode: str) -> bool:
    if mode == "no":
        return False
    if mode == "yes":
        return True
    # auto
    edge_density = compute_edge_density(pil_img)
    width, height = pil_img.size
    # Trigger OCR for likely text-like images
    if width < 32 or height < 32:
        return False
    return edge_density >= 0.010


def run_tesseract_ocr(pil_img: Image.Image, lang: str) -> Optional[str]:
    try:
        import pytesseract  # lazy import

        config = "--oem 1 --psm 6"
        txt = pytesseract.image_to_string(pil_img, lang=lang, config=config)
        return normalize_text(txt)
    except Exception as exc:  # pragma: no cover - tesseract optional
        logging.warning("OCR failed or unavailable: %s", exc)
        return None


def image_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def generate_caption_openai(
    pil_img: Image.Image,
    model: str,
    api_key: str,
    base_url: Optional[str],
) -> Optional[str]:
    try:
        import requests  # lazy

        url = (base_url or "https://api.openai.com/v1") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        b64 = image_to_base64(pil_img)
        prompt = (
            "Provide one short, factual caption (<= 160 chars). "
            "Avoid subjective adjectives and opinions."
        )
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise captioning assistant.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": b64}},
                    ],
                },
            ],
            "temperature": 0.2,
            "max_tokens": 60,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        return normalize_text(text)
    except Exception as exc:  # pragma: no cover - network optional
        logging.warning("OpenAI caption failed: %s", exc)
        return None


def generate_caption_blip(pil_img: Image.Image) -> Optional[str]:
    try:
        from transformers import pipeline  # lazy

        pipe = pipeline(
            task="image-to-text",
            model="Salesforce/blip-image-captioning-base",
        )
        outputs = pipe(pil_img)
        if not outputs:
            return None
        text = outputs[0].get("generated_text") or outputs[0].get("text")
        return normalize_text(text or "")
    except Exception as exc:  # pragma: no cover - optional heavy dep
        logging.warning("BLIP caption failed or unavailable: %s", exc)
        return None


def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------- Core Extraction ----------


@dataclass
class Row:
    doc_id: str
    page: int
    unit: str  # block|image|page_summary
    section: Optional[str]
    text: Optional[str]
    spans: Optional[List[Dict[str, Any]]]
    image_ref: Optional[str]
    image_bbox: Optional[List[float]]
    ocr_text: Optional[str]
    caption: Optional[str]
    embedding_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "page": self.page,
            "unit": self.unit,
            "section": self.section,
            "text": self.text,
            "spans": self.spans,
            "image_ref": self.image_ref,
            "image_bbox": self.image_bbox,
            "ocr_text": self.ocr_text,
            "caption": self.caption,
            "embedding_id": self.embedding_id,
        }


def extract_page(
    doc: fitz.Document,
    page_index_zero: int,
    doc_id: str,
    images_dir: Path,
    ocr_mode: str,
    ocr_lang: str,
    caption_provider: str,
    caption_model: str,
    openai_cfg: OpenAIConfig,
    min_caption_len: int,
) -> Tuple[List[Row], List[Tuple[int, str]]]:
    page = doc.load_page(page_index_zero)
    page_num = page_index_zero + 1
    page_dict = page.get_text("dict")
    blocks = page_dict.get("blocks", [])

    # Compute font size percentile for heading detection
    span_sizes: List[float] = []
    for blk in blocks:
        if blk.get("type", 0) != 0:
            continue
        for line in blk.get("lines", []):
            for span in line.get("spans", []):
                size = float(span.get("size", 0))
                if size > 0:
                    span_sizes.append(size)
    heading_threshold = percentile(span_sizes, 85.0) if span_sizes else 0.0

    rows: List[Row] = []
    embeds: List[Tuple[int, str]] = []  # (row_index_in_rows, text)

    current_section: Optional[str] = None
    page_text_parts: List[str] = []
    image_counter = 0

    for blk in blocks:
        btype = blk.get("type", 0)
        bbox = [float(x) for x in blk.get("bbox", [])]
        if btype == 0:  # text
            spans_out: List[Dict[str, Any]] = []
            texts: List[str] = []
            max_block_size = 0.0
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    if span_text:
                        texts.append(span_text)
                    font_name = span.get("font", "")
                    font_size = float(span.get("size", 0.0))
                    max_block_size = max(max_block_size, font_size)
                    spans_out.append(
                        {
                            "bbox": [float(x) for x in span.get("bbox", [])] if span.get("bbox") else None,
                            "font_size": font_size,
                            "bold": guess_bold(font_name),
                        }
                    )
            block_text = normalize_text(" ".join(texts))
            if not block_text:
                continue
            page_text_parts.append(block_text)

            # Heading detection - update current section if block looks like heading
            if max_block_size >= heading_threshold and len(block_text) <= 80:
                current_section = block_text

            row = Row(
                doc_id=doc_id,
                page=page_num,
                unit="block",
                section=current_section,
                text=block_text,
                spans=spans_out,
                image_ref=None,
                image_bbox=None,
                ocr_text=None,
                caption=None,
                embedding_id=None,
            )
            rows.append(row)
            embeds.append((len(rows) - 1, block_text))
        elif btype == 1:  # image block
            xref = blk.get("image")
            if xref is None:
                continue
            try:
                img_info = doc.extract_image(xref)
                img_bytes = img_info["image"]
                with Image.open(io.BytesIO(img_bytes)) as pi:
                    pil_img = pi.convert("RGB")
            except Exception:
                # Fallback: render area as pixmap
                pix = page.get_pixmap(clip=fitz.Rect(bbox))
                pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            image_counter += 1
            image_name = f"page{page_num}_img{image_counter}.png"
            image_path = images_dir / image_name
            pil_img.save(image_path)

            ocr_text: Optional[str] = None
            if should_ocr(pil_img, ocr_mode):
                ocr_text = run_tesseract_ocr(pil_img, ocr_lang)

            caption: Optional[str] = None
            if caption_provider == "openai":
                api_key = os.getenv(openai_cfg.api_key_env, "")
                if api_key:
                    caption = generate_caption_openai(pil_img, caption_model, api_key, openai_cfg.base_url)
                else:  # pragma: no cover - env may be missing in CI
                    logging.warning("OPENAI API key not set; skipping captions")
            elif caption_provider == "blip":
                caption = generate_caption_blip(pil_img)
            # fallback to short OCR summary if caption missing
            if (not caption or len(caption) < min_caption_len) and ocr_text:
                caption = normalize_text(ocr_text[:160])

            row = Row(
                doc_id=doc_id,
                page=page_num,
                unit="image",
                section=current_section,
                text=None,
                spans=None,
                image_ref=str(Path("images") / image_name),
                image_bbox=[float(v) for v in bbox] if bbox else None,
                ocr_text=ocr_text,
                caption=caption,
                embedding_id=None,
            )
            rows.append(row)

            # Embedding text for image = caption + ocr
            embed_text_parts = [p for p in [caption, ocr_text] if p]
            embed_text = normalize_text(". ".join(embed_text_parts)) if embed_text_parts else ""
            if embed_text:
                embeds.append((len(rows) - 1, embed_text))

    # Page summary row
    page_text_joined = normalize_text(" \n ".join(page_text_parts)) if page_text_parts else None
    if page_text_joined:
        rows.append(
            Row(
                doc_id=doc_id,
                page=page_num,
                unit="page_summary",
                section=None,
                text=page_text_joined,
                spans=None,
                image_ref=None,
                image_bbox=None,
                ocr_text=None,
                caption=None,
                embedding_id=None,
            )
        )

    return rows, embeds


# ---------- Embeddings ----------


def compute_embeddings_openai(
    texts: List[str], model: str, api_key: str, base_url: Optional[str]
) -> np.ndarray:
    try:
        import requests

        url = (base_url or "https://api.openai.com/v1") + "/embeddings"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        vectors: List[List[float]] = []
        for batch in batched(texts, 128):
            payload = {"model": model, "input": batch}
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # Order is preserved
            for item in data.get("data", []):
                vectors.append(item.get("embedding", []))
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return np.array(vectors, dtype=np.float32)
    except Exception as exc:  # pragma: no cover - network optional
        logging.warning("OpenAI embeddings failed: %s", exc)
        return np.zeros((0, 0), dtype=np.float32)


def compute_embeddings_hf(texts: List[str], model_name: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        vecs = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
        return vecs.astype(np.float32)
    except Exception as exc:  # pragma: no cover - optional heavy dep
        logging.warning("HF embeddings failed or unavailable: %s", exc)
        return np.zeros((0, 0), dtype=np.float32)


def write_embeddings(out_dir: Path, ids: List[str], vectors: np.ndarray) -> None:
    if vectors.size == 0 or not ids:
        return
    embeds_dir = out_dir / "embeddings"
    embeds_dir.mkdir(parents=True, exist_ok=True)
    vec_path = embeds_dir / "vectors.npy"
    index_path = embeds_dir / "index.json"
    np.save(vec_path, vectors)
    index = {
        "vector_dim": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "ids": ids,
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert PDFs into a multimodal corpus (text + images + OCR + captions + embeddings)",
        add_help=True,
    )
    p.add_argument("--pdf", required=True, help="Path to PDF file")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--format", default="jsonl", choices=["jsonl", "yaml", "markdown", "md"], help="Secondary output format (JSONL always produced)")
    p.add_argument("--ocr", default="auto", choices=["auto", "yes", "no"], help="OCR mode for images")
    p.add_argument("--caption-provider", default="none", choices=["openai", "blip", "none"], help="Image captioning provider")
    p.add_argument("--embed-provider", default="none", choices=["openai", "hf", "none"], help="Embedding provider")
    p.add_argument("--max-pages", type=int, default=0, help="Max pages to process (0 = all)")
    p.add_argument("--min-cap-len", type=int, default=6, help="Minimum caption length before falling back to OCR text")
    p.add_argument("--lang", default=None, help="Language code for OCR (e.g., eng, deu)")
    p.add_argument("--config", default=None, help="YAML config path")
    p.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    # If started without arguments, print help and exit gracefully
    if len(sys.argv) == 1:
        p.print_help(sys.stderr)
        sys.exit(1)
    return p.parse_args()


def load_config(cli_args: argparse.Namespace) -> RuntimeConfig:
    cfg = RuntimeConfig()
    # YAML overrides
    if cli_args.config:
        import yaml  # lazy

        with open(cli_args.config, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = RuntimeConfig(**data)
    # CLI overrides
    if cli_args.lang:
        cfg.language = cli_args.lang
    cfg.caption_provider = cli_args.caption_provider
    cfg.embed_provider = cli_args.embed_provider
    cfg.max_pages = cli_args.max_pages
    cfg.min_caption_len = cli_args.min_cap_len
    cfg.ocr = cli_args.ocr
    return cfg


def maybe_secondary_outputs(fmt: str) -> Tuple[bool, bool]:
    want_yaml = fmt == "yaml"
    want_md = fmt in {"markdown", "md"}
    return want_yaml, want_md


def main() -> int:
    args = parse_args()
    setup_logger(args.quiet)
    load_dotenv()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = ensure_out_dirs(out_dir)
    want_yaml, want_md = maybe_secondary_outputs(args.format)

    if not pdf_path.exists():
        logging.error("PDF not found: %s", pdf_path)
        return 2

    cfg = load_config(args)
    doc_id = pdf_path.stem

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logging.error("Failed to open PDF: %s", exc)
        return 3

    max_pages = cfg.max_pages if cfg.max_pages and cfg.max_pages > 0 else doc.page_count
    rows: List[Row] = []
    embeds_for_rows: List[Tuple[int, str]] = []

    logging.info("Processing %s pages", max_pages)
    for page_idx in tqdm(range(min(max_pages, doc.page_count)), disable=args.quiet):
        page_rows, page_embeds = extract_page(
            doc=doc,
            page_index_zero=page_idx,
            doc_id=doc_id,
            images_dir=dirs["images"],
            ocr_mode=cfg.ocr,
            ocr_lang=cfg.language,
            caption_provider=cfg.caption_provider,
            caption_model=cfg.caption_model,
            openai_cfg=cfg.openai,
            min_caption_len=cfg.min_caption_len,
        )
        embeds_offset = len(rows)
        rows.extend(page_rows)
        # Adjust embed row indices relative to global rows
        for local_row_idx, text in page_embeds:
            embeds_for_rows.append((embeds_offset + local_row_idx, text))

    # Assign embedding ids and compute vectors
    embedding_ids: List[str] = []
    embedding_texts: List[str] = []
    for _, text in embeds_for_rows:
        embedding_ids.append(f"e_{len(embedding_ids):06d}")
        embedding_texts.append(text)
    for (row_index, _), emb_id in zip(embeds_for_rows, embedding_ids):
        rows[row_index].embedding_id = emb_id

    vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
    if embedding_texts and cfg.embed_provider != "none":
        if cfg.embed_provider == "openai":
            api_key = os.getenv(cfg.openai.api_key_env, "")
            if not api_key:  # pragma: no cover - env may be missing in CI
                logging.warning("OPENAI API key not set; skipping embeddings")
            else:
                vectors = compute_embeddings_openai(embedding_texts, cfg.embed_model, api_key, cfg.openai.base_url)
        elif cfg.embed_provider == "hf":
            vectors = compute_embeddings_hf(embedding_texts, cfg.embed_model)

        if vectors.size > 0:
            write_embeddings(out_dir, embedding_ids, vectors)

    # Always write JSONL
    jsonl_path = out_dir / "data.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    if want_yaml:
        try:
            import yaml

            yaml_path = out_dir / "data.yaml"
            with yaml_path.open("w", encoding="utf-8") as f:
                # Write as a YAML list of rows
                yaml.safe_dump([r.to_dict() for r in rows], f, sort_keys=False, allow_unicode=True)
        except Exception as exc:  # pragma: no cover - yaml should exist
            logging.warning("Failed to write YAML: %s", exc)

    if want_md:
        md_path = out_dir / "notes.md"
        try:
            total_images = len([r for r in rows if r.unit == "image"])
            total_blocks = len([r for r in rows if r.unit == "block"])
            lines = [
                f"# Document {doc_id}",
                "",
                f"Pages processed: {min(max_pages, doc.page_count)}",
                f"Text blocks: {total_blocks}",
                f"Images: {total_images}",
            ]
            md_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            logging.warning("Failed to write Markdown: %s", exc)

    logging.info("Done. Wrote %s", jsonl_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

