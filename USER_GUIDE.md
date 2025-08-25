## User Guide: PDF2AI – PDF to AI-Ready Corpus Converter

### Overview
PDF2AI converts PDFs into an AI-agent friendly corpus. It extracts per-page text blocks, images, OCR text, concise image captions, optional vision summaries, and embeddings. Use the JSONL for programmatic agents; Markdown is for human review.

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
- BLIP will log which device it's using. Models are cached after first run.

### Running
CLI example:
```bash
python pdf2ai.py \
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

### AI Agent Integration Guide

#### Format Selection for Different Use Cases

**JSONL Format (Recommended for AI Agents)**
- **Best for**: Programmatic processing, semantic search, data extraction
- **Structure**: Each line is a JSON object with structured fields
- **Advantages**: 
  - Semantic sections with `section` field
  - Embedding IDs for vector search
  - Bounding box coordinates for spatial analysis
  - Deterministic ordering for reproducible results

**YAML Format**
- **Best for**: Configuration, hierarchical data extraction, rule-based systems
- **Structure**: Hierarchical document structure
- **Advantages**:
  - Easy parsing for configuration systems
  - Human-readable structure
  - Good for document analysis workflows

**Markdown Format**
- **Best for**: Human review, documentation, content summarization
- **Structure**: Readable text with section headers
- **Advantages**:
  - Easy to read and navigate
  - Good for content summarization
  - Section-based navigation

#### Example AI Agent Workflows

**1. Document Analysis Pipeline**
```python
import json
from typing import Dict, List

def analyze_document(jsonl_path: str) -> Dict[str, List[str]]:
    """Extract structured information from PDF corpus."""
    sections = {}
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            row = json.loads(line)
            section = row.get('section', 'general')
            if section not in sections:
                sections[section] = []
            sections[section].append(row['text'])
    
    return sections

# Usage
document_sections = analyze_document('out/document.jsonl')
for section, content in document_sections.items():
    print(f"Section: {section}")
    print(f"Content blocks: {len(content)}")
```

**2. Semantic Search Implementation**
```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticSearcher:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = {}
        self.texts = {}
    
    def load_corpus(self, jsonl_path: str):
        """Load document corpus with embeddings."""
        with open(jsonl_path, 'r') as f:
            for line in f:
                row = json.loads(line)
                if row.get('embedding_id') and row.get('text'):
                    self.texts[row['embedding_id']] = row['text']
        
        # Generate embeddings
        texts_list = list(self.texts.values())
        embeddings = self.model.encode(texts_list)
        
        for i, emb_id in enumerate(self.texts.keys()):
            self.embeddings[emb_id] = embeddings[i]
    
    def search(self, query: str, top_k: int = 5):
        """Search for similar content."""
        query_embedding = self.model.encode([query])
        
        similarities = []
        for emb_id, embedding in self.embeddings.items():
            similarity = np.dot(query_embedding[0], embedding)
            similarities.append((emb_id, similarity, self.texts[emb_id]))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Usage
searcher = SemanticSearcher()
searcher.load_corpus('out/document.jsonl')
results = searcher.search("trading strategy implementation")
```

**3. Section-Based Content Extraction**
```python
def extract_specific_sections(jsonl_path: str, target_sections: List[str]) -> Dict[str, str]:
    """Extract content from specific document sections."""
    section_content = {}
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            row = json.loads(line)
            section = row.get('section')
            if section in target_sections:
                if section not in section_content:
                    section_content[section] = []
                section_content[section].append(row['text'])
    
    # Combine section content
    return {section: ' '.join(content) for section, content in section_content.items()}

# Usage
trading_sections = extract_specific_sections(
    'out/document.jsonl', 
    ['Trading System', 'Risk Management', 'Entry Rules']
)
```

#### Best Practices for AI Agent Development

1. **Use JSONL for Production**: Most efficient for programmatic processing
2. **Leverage Embeddings**: Use `embedding_id` for semantic search capabilities
3. **Section-Based Processing**: Use `section` field for targeted content extraction
4. **Spatial Analysis**: Use bounding box coordinates for layout-aware processing
5. **Error Handling**: Always check for missing fields in JSON objects

#### Performance Optimization

- **Batch Processing**: Process multiple documents in batches
- **Caching**: Cache embeddings and processed results
- **Parallel Processing**: Use multiprocessing for large document sets
- **Memory Management**: Stream large JSONL files instead of loading entirely

### Tips for Better Results
- Set Caption=openai for higher-quality visual descriptions.
- Use --ocr yes for small-text images.
- Enable page_vision for layout summaries when you need agent-readable structure.
- Use JSONL format for AI agent integration and semantic search.
- Leverage section-based extraction for targeted content analysis.

