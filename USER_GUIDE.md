## User Guide: pdf2mm

### Overview
pdf2mm converts PDFs into an AI-agent friendly corpus. It extracts per-page text blocks, images, OCR text, concise image captions, optional vision summaries, and embeddings. Use the JSONL for programmatic agents; Markdown is for human review.

### Installation
1) Create venv and install deps
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) System tools (optional)
- Tesseract for OCR: apt/brew/choco install tesseract
- Poppler utils: apt/brew install poppler

### GPU Acceleration
BLIP (captions) and sentence-transformers (embeddings) can use GPU.

- Windows/Linux + NVIDIA (CUDA)
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- macOS (Apple Silicon, MPS)
```bash
pip install torch torchvision
python -c "import torch; print(hasattr(torch.backends,'mps') and torch.backends.mps.is_available())"
```

Notes:
- Ensure latest GPU drivers are installed.
- BLIP will log which device it’s using. Models are cached after first run.

### Running
CLI example:
```bash
python pdf2mm.py \
  --pdf path/to/file.pdf \
  --outdir out \
  --format jsonl \
  --ocr auto \
  --caption-provider blip \
  --embed-provider hf
```

GUI:
```bash
./run_gui.sh  # or run_gui.bat on Windows
```

### Providers
- Captions: none | openai | blip
- Embeddings: none | openai | hf
- Vision: page_vision and image_vision can be enabled with openai.

### Azure OpenAI
Set these in config (YAML or JSON editor from GUI):
```json
{
  "openai": {
    "api_key_env": "OPENAI_API_KEY",
    "azure_use": true,
    "azure_endpoint": "https://<resource>.openai.azure.com",
    "azure_deployment": "<chat_or_vision_deployment>",
    "azure_embed_deployment": "<embedding_deployment>",
    "azure_api_version": "2024-02-15-preview"
  }
}
```

### Outputs
- JSONL: out/<basename>.jsonl
- YAML (optional): out/<basename>.yaml
- Markdown (optional): out/<basename>.md
- Images: out/images/
- Embeddings: out/embeddings/<basename>.*
- New units: page_vision and image (with visual JSON in text field when enabled)

### Tips for Better Results
- Set Caption=openai for higher-quality visual descriptions.
- Use --ocr yes for small-text images.
- Enable page_vision for layout summaries when you need agent-readable structure.

