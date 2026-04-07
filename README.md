# Cognitive Bridge — Patient-Centric Medical Jargon Simplifier

A privacy-first Chrome extension that simplifies complex medical jargon into layperson terms in real-time. All processing happens locally in the browser using Transformers.js — zero data ever leaves the user's device.

## Architecture

```
Patient Portal Page
        |
   MutationObserver (detects medical content)
        |
   NER Model (DistilBERT / ONNX)
        |
   Confidence Threshold
      /       \
  HIGH          LOW
    |             |
T5 Simplify   MedlinePlus Glossary
    \             /
   Shadow DOM Tooltip
        |
   Patient Portal Page (with highlights & tooltips)
```

**Key features:**
- Local-first inference via Transformers.js — zero network requests during processing
- DistilBERT NER model fine-tuned on BC5CDR for Disease & Chemical entity recognition
- T5 simplification model for context-aware sentence rewriting
- MedlinePlus glossary fallback for low-confidence entities
- Shadow DOM tooltip isolation — no CSS conflicts with host pages
- Manifest V3 compliant Chrome extension

## Project Structure

```
├── extension/              Chrome extension source
│   ├── manifest.json       Manifest V3 configuration
│   ├── background/         Service worker
│   ├── content/            Content scripts (observer, pipeline, tooltips)
│   ├── popup/              Extension popup UI
│   ├── data/               MedlinePlus glossary JSON
│   └── icons/              Extension icons
├── training/               Model training & export
│   ├── ner/                NER fine-tuning + ONNX export
│   ├── simplification/     T5 fine-tuning + ONNX export
│   └── data/               Data preparation scripts
├── evaluation/             Evaluation metrics & baselines
├── demo/                   Mock patient portal for testing
├── scripts/                Utility scripts
├── webpack.config.js       Build configuration
└── package.json            Node.js dependencies
```

## Quick Start

### Prerequisites

- Node.js 18+
- npm 9+
- Python 3.9+ (for training/evaluation only)
- Google Chrome

### 1. Install Dependencies

```bash
npm install --ignore-scripts
```

> Note: `--ignore-scripts` skips building the `sharp` native module, which is an optional dependency of Transformers.js not needed for browser-based inference.

### 2. Build the Extension

```bash
npm run build
```

This creates a `dist/` folder with the bundled extension.

### 3. Load in Chrome

1. Open Chrome and navigate to `chrome://extensions`
2. Enable **Developer mode** (toggle in the top right)
3. Click **Load unpacked**
4. Select the `dist/` folder from this project
5. The Cognitive Bridge icon should appear in your toolbar

### 4. Test with Demo Page

Open `demo/index.html` in Chrome. The extension will automatically detect medical terms, highlight them, and provide simplified explanations on hover.

## Development

```bash
# Watch mode for development
npm run dev

# Clean build artifacts
npm run clean
```

## Model Training (Optional)

If you want to fine-tune the models from scratch:

### Setup Python Environment

```bash
pip install -r training/requirements.txt
```

### NER Model (DistilBERT on BC5CDR)

```bash
# 1. Prepare the BC5CDR dataset
python training/data/prepare_bc5cdr.py

# 2. Fine-tune DistilBERT
python training/ner/finetune_ner.py

# 3. Export to ONNX with 8-bit quantization
python training/ner/export_onnx_ner.py
```

### Simplification Model (T5-small)

```bash
# 1. Prepare simplification pairs
python training/data/prepare_simplification.py

# 2. Fine-tune T5-small
python training/simplification/finetune_t5.py

# 3. Export to ONNX with 8-bit quantization
python training/simplification/export_onnx_t5.py
```

### Update Glossary

```bash
python scripts/build_glossary.py
```

## Evaluation

Run evaluation metrics (Flesch-Kincaid Grade Level, SARI, Entity Preservation Rate):

```bash
# Prepare test data first
python training/data/prepare_simplification.py

# Run evaluation
python evaluation/evaluate.py

# Run baseline comparisons
python evaluation/run_baselines.py
```

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Flesch-Kincaid Grade Level** | Readability score (lower = easier) | ≤ 8th grade |
| **SARI** | Simplification quality (0-100) | Higher is better |
| **Entity Preservation Rate** | Fraction of medical entities retained | > 90% |

### Baselines

1. **Glossary-only**: No rewriting, just tooltip definitions
2. **Vanilla T5-small**: Off-the-shelf T5 without medical fine-tuning
3. **Cognitive Bridge**: Our fine-tuned system with confidence routing

## Privacy Guarantee

- All NLP inference runs locally in the browser via WebAssembly
- Zero network requests are made during text processing
- No patient data is sent to any external server
- Verify by opening Chrome DevTools Network tab during use

## Technical Details

- **NER Model**: DistilBERT fine-tuned on BC5CDR (Disease + Chemical entities)
- **Simplification Model**: T5-small fine-tuned on medical simplification pairs
- **Runtime**: Transformers.js with ONNX Runtime Web (WebAssembly backend)
- **Quantization**: 8-bit dynamic quantization (~50MB total footprint)
- **UI Isolation**: Shadow DOM prevents CSS conflicts with host pages
- **Extension Standard**: Chrome Manifest V3

## Team

- **Yonghee Han** — Bio-NER and simplification models, fine-tuning, evaluation
- **Vinay Singh** — Extension infrastructure, ONNX quantization, MV3 packaging
- **Shanti Patel** — UI/UX design, Shadow DOM, dataset curation

## References

- Li et al. (2016). BioCreative V CDR Task Corpus
- Raffel et al. (2020). T5: Exploring the Limits of Transfer Learning
- Vora et al. (2025). S-3 Pipeline for Biomedical Text Simplification
- Xenova (2023). Transformers.js
- Xu et al. (2016). Optimizing Statistical Machine Translation for Text Simplification
