## PDF2AI – PDF to AI-Ready Corpus Converter

Compact, production-ready CLI that converts PDFs into AI-agent-friendly corpora. It extracts per-page text blocks with bounding boxes, exports images, runs optional OCR and image captioning, and builds optional embeddings for semantic search. Outputs JSONL by default, with optional YAML and Markdown summaries.

### Features
- **Structured text**: PyMuPDF-based extraction with block bboxes, simple heading detection and section assignment.
- **Images**: Exported as PNG with page and index references.
- **OCR**: Tesseract optional; auto/yes/no modes with a quick gating heuristic.
- **Captions**: OpenAI (remote) or BLIP (local transformers) with short factual sentences.
- **Embeddings**: OpenAI or sentence-transformers (`all-MiniLM-L6-v2`).
- **Deterministic IDs**: Stable `embedding_id` ordering.
- **AI Agent Optimization**: Structured output formats optimized for programmatic processing.

### Quickstart

POSIX:
```bash
./run.sh --pdf ./samples/sample.pdf --outdir out --format jsonl --ocr auto --caption-provider none --embed-provider hf
```

Windows:
```bat
run.bat --pdf .\samples\sample.pdf --outdir out --format yaml --caption-provider openai --embed-provider none
```

GUI (optional, Tkinter):
```bash
python gui.py
```
Select a PDF, configure options, preview PDF pages and output text, then click Run.

Starters:
- POSIX: `./run_gui.sh`
- Windows: `run_gui.bat`

### Install system dependencies

- **Tesseract OCR** (optional, for `--ocr auto|yes`)
  - Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`
  - macOS (Homebrew): `brew install tesseract`
  - Windows:
    - Installer: `https://github.com/tesseract-ocr/tesseract`
    - Or Chocolatey: `choco install tesseract`

- **Poppler utils** (optional, helpful for troubleshooting):
  - Debian/Ubuntu: `sudo apt-get install -y poppler-utils`
  - macOS (Homebrew): `brew install poppler`
  - Windows: `https://github.com/oschwartz10612/poppler-windows`

Python dependencies are handled by the `run.sh` / `run.bat` scripts via a local `.venv`.

### Configuration

Runtime config via YAML (optional): `configs/example.yaml`
```yaml
language: "de"
caption_provider: "openai"
caption_model: "gpt-4o-mini"
embed_provider: "hf"
embed_model: "sentence-transformers/all-MiniLM-L6-v2"
max_pages: 0
min_caption_len: 6
ocr: "auto"
openai:
  api_key_env: "OPENAI_API_KEY"
  base_url: null
```

Environment variables go in `.env` (see `.env.example`).

### CLI

```bash
python pdf2ai.py \
  --pdf path/to/file.pdf \
  --outdir out_dir \
  --format jsonl \
  --ocr auto \
  --caption-provider openai|blip|none \
  --embed-provider openai|hf|none \
  --max-pages 0 \
  --min-cap-len 6 \
  --lang de \
  --config configs/example.yaml
```

### Output

Directory structure example (basename = sample):
```
out/
  sample.jsonl
  sample.yaml           # if --format yaml
  sample.md             # if --format markdown|md (full document)
  images/
    page1_img1.png
    page2_img1.png
  embeddings/
    sample.index.json
    sample.vectors.npy
```

JSONL schema per row:
```json
{
  "doc_id": "basename_without_ext",
  "page": 16,
  "unit": "block|image|page_summary",
  "section": "string|null",
  "text": "normalized block text or page text",
  "spans": [{"bbox":[x0,y0,x1,y1],"font_size":N,"bold":bool}],
  "image_ref": "images/page16_img2.png",
  "image_bbox": [x0,y0,x1,y1],
  "ocr_text": "string|null",
  "caption": "short factual caption|null",
  "embedding_id": "e_000123|null"
}
```

### AI Agent Usage Guide

**Best Practices for AI Agent Integration:**

1. **JSONL Format (Recommended)**: Use for programmatic processing and semantic search
   - Structured data with semantic sections
   - Embedding IDs for vector search
   - Bounding box coordinates for spatial analysis

2. **YAML Format**: Use for configuration and structured data extraction
   - Hierarchical document structure
   - Easy parsing for rule-based systems

3. **Markdown Format**: Use for human review and documentation
   - Readable summaries
   - Section-based navigation

**Example AI Agent Workflow:**
```python
import json

# Load JSONL for semantic search
with open('out/document.jsonl', 'r') as f:
    for line in f:
        row = json.loads(line)
        if row['section'] == 'target_section':
            # Process specific section
            process_section(row['text'])
```

**Semantic Search Example:**
```python
# Use embedding_id for vector similarity search
embedding_id = "e_000123"
# Query vector database for similar content
similar_content = vector_db.search(embedding_id)
```

Notes:
- Reading order preserved from PyMuPDF blocks. Headings inferred via top font-size percentile per page.
- OCR gating uses a quick edge-density heuristic; `--ocr yes` forces, `--ocr no` disables.
- Captioning returns at most one short factual sentence. BLIP uses `Salesforce/blip-image-captioning-base` if installed.
- Embedding text for images is `caption + ocr_text`; for text blocks it's the block text.
- Embeddings are batched to reduce API calls. Respect provider rate limits.

### Providers

- **OpenAI**
  - Set `OPENAI_API_KEY` in environment or `.env`.
  - Optional `OPENAI_BASE_URL` for compatible gateways.
  - Configure model names in YAML.

- **Hugging Face / sentence-transformers**
  - Install `sentence-transformers` if using `--embed-provider hf`.
  - Default: `sentence-transformers/all-MiniLM-L6-v2`.

- **BLIP captions**
  - Install `transformers` to enable `--caption-provider blip`.

### Development

Validate code quality:
```bash
python -m pyflakes pdf2ai.py
```

See the full user guide (GPU setup, Azure config, GUI usage): [USER_GUIDE.md](USER_GUIDE.md)

Run tests:
```bash
pytest -q
```

### Author

**Andre Lorbach**  
Email: [alorbach@adiscon.com](mailto:alorbach@adiscon.com)  
Software Developer since 1999

This code was created using AI Agent Models and human code oversight.

The goal is to create a utility that can convert PDF Documents with images to a better understandable version, suited to work better for AI Agent Tasks.

### License

MIT (see `LICENSE`).

