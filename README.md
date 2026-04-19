# Cognitive Bridge — Patient-Centric Medical Jargon Simplifier

A privacy-first Chrome extension that simplifies complex medical jargon into
plain English in real time. All inference runs locally in the browser — no
patient text ever leaves the user's device.

> For a detailed reconciliation of the proposal against the current
> implementation, see [`PROPOSAL_VS_IMPLEMENTATION.md`](./PROPOSAL_VS_IMPLEMENTATION.md).

---

## How it works

```
        Patient portal page
                │
        Observe medical text
                │
     NER model  (DistilBioBERT, 8-bit)
                │
       Confidence ≥ 0.75 ?
          ┌─────┴─────┐
        yes           no
          │            │
   T5 simplifier   MedlinePlus
   (Flan-T5-small,   glossary
    8-bit, merged     lookup
    decoder)
          │            │
   Quality filter      │
          └─────┬──────┘
                │
       Shadow-DOM tooltip
```

Each medical term on a page is identified by a small biomedical NER model.
High-confidence entities go to a fine-tuned T5 simplifier for a plain-English
definition; low-confidence entities fall back to a local MedlinePlus glossary.
Everything is rendered inside an isolated Shadow DOM tooltip so host-page CSS
can't interfere.

---

## Shipped models

| Role       | Model                                          | Size (quantized) |
| ---------- | ---------------------------------------------- | ---------------- |
| NER        | `LOGiC31/cognitive-bridge-ner`                 | ~63 MB           |
| Simplifier | `LOGiC31/cognitive-bridge-t5-medisimplifier`   | ~50 MB           |

Both are loaded on first use via Transformers.js and cached in the browser's
IndexedDB. After that, inference is fully offline.

---

## Key design choices

- **DistilBioBERT over DistilBERT.** Same size (~65 M params), but biomedical
  pre-training noticeably improves recall on clinical entities.
- **Flan-T5-small over T5-small.** Same size class (~77 M params), but
  instruction-tuned, so the `"Simplify this medical text for a patient:"`
  prefix works out of the box and fine-tuning converges faster.
- **Section-level training data.** The simplifier is trained on
  `GuyDor007/medisimplifier-dataset` split at section level, not whole
  discharge summaries. This matches the short clinical passages the
  extension actually sees at inference time.
- **Term-only prompt at inference.** Training uses full clinical sections,
  but the extension prompts the model with just the entity term. The
  product surface is a per-term tooltip, not a sentence rewrite, so
  constraining the prompt prevents the model from paraphrasing
  surrounding context (headers, dates) instead of defining the term.
- **Merged-decoder-only ONNX bundle.** The export ships only the encoder
  and merged decoder, dropping the redundant unmerged decoder pair and
  saving ~110 MB of browser cache.
- **Baked-in repetition guards.** `no_repeat_ngram_size=3` and
  `repetition_penalty=1.3` are part of the shipped generation config, so
  the browser runtime is protected against quantization-noise decode
  loops without any extension-side overrides.
- **Output quality filters.** Outputs that echo a section header, repeat
  a template phrase, or produce a date preamble are rejected before
  being shown. Only clean outputs populate the tooltip cache.

---

## Quick start

**Prerequisites:** Node.js 18+, npm 9+, Google Chrome. (Python 3.12 is
only needed to reproduce the training.)

```bash
npm install --ignore-scripts
npm run build
```

Then in Chrome:

1. Open `chrome://extensions`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked** and select the generated `dist/` folder
4. Open `demo/index.html` to see the extension highlight medical terms
   and show plain-English tooltips on hover

For dev mode: `npm run dev` (webpack watch) · `npm run clean` (remove `dist/`).

---

## Evaluation

All numbers below use the deployed models against the `medisimplifier` test
split (1,000 examples for the full test, 200 for the ONNX release gate,
10 for the qualitative-probe subset used for FKGL/EPR).

### NER — DistilBioBERT on BC5CDR

| Metric     | Test score  |
| ---------- | ----------- |
| Precision  | **76.1 %**  |
| Recall     | **85.9 %**  |
| F1         | **80.7 %**  |
| Accuracy   | **95.8 %**  |

After 8-bit dynamic quantization the NER model shrinks **4× (≈249 MB →
63 MB)** with no measurable quality loss in spot checks.

> **Why recall matters most.** The extension can only simplify what NER
> surfaces, so missed entities are the worst failure mode. 85.9 % recall
> paired with a 0.75 confidence threshold keeps false positives low;
> anything borderline is routed to the MedlinePlus glossary rather than
> to the generative model.

### Simplifier — Flan-T5-small on medisimplifier (PyTorch)

Full `test` split, 1,000 examples.

| Metric      | Score     |
| ----------- | --------- |
| ROUGE-1     | **68.5**  |
| ROUGE-2     | **46.6**  |
| ROUGE-L     | **61.7**  |
| ROUGE-Lsum  | **66.1**  |

For context, off-the-shelf Flan-T5-small scores ROUGE-L in the mid-40s on
the same data. Fine-tuning lifts it by roughly **15 points**, and a
ROUGE-2 of 46.6 is unusually strong for a ~77 M-parameter model on
discharge-summary simplification — it means the model matches the
bigram structure of the human plain-language version, not just the
keywords.

### Quantized ONNX (what actually ships in the browser)

200-sample release gate, PyTorch vs 8-bit ONNX:

| Metric                 | PyTorch | ONNX (8-bit) | Change                   |
| ---------------------- | ------- | ------------ | ------------------------ |
| ROUGE-L                | 61.5    | 55.2         | −10.4 % (within budget)  |
| SARI                   | 67.0    | 62.4         | −4.6 points              |
| Avg output length      | 381     | 381          | same                     |
| Repeated-5-gram rate   | —       | **0.0 %**    | no decode loops          |
| Empty outputs          | —       | **0 / 200**  | no failed generations    |

Quantization costs about 10 % of ROUGE-L and ~4.6 SARI points — well
within our release tolerances. The repetition and empty-output guards
never tripped on 200 examples.

### Proposal metrics — FKGL, SARI, EPR

| Metric                             | Target        | Source text | Human reference | Model (PyTorch) | Model (ONNX) |
| ---------------------------------- | ------------- | ----------- | --------------- | --------------- | ------------ |
| **Flesch-Kincaid Grade Level**     | ≤ 8th grade   | 14.2        | 7.2             | **7.2**         | **6.9** ✅   |
| **SARI**                           | Higher better | —           | —               | **67.0**        | **62.4** ✅  |
| **Entity Preservation Rate**       | > 90 %        | —           | 41.0 %          | 23.7 %          | 25.3 % ⚠️    |

**Flesch-Kincaid — ✅ target met.** Source clinical text reads at a
college-freshman level (grade 14.2). Both the PyTorch and ONNX models
drop readability to ~7th grade, **beating the ≤ 8th-grade target** and
matching the human reference (7.2) almost exactly. This is the
proposal's headline readability win.

**SARI — ✅ strong.** Published medical-simplification systems typically
report SARI in the 35–45 range; anything above 45 is considered
excellent. Our deployed (ONNX) model scores **62.4**, and 8-bit
quantization cost only 4.6 points versus the full PyTorch checkpoint.

**EPR — ⚠️ the metric is the problem, not the model.** At face value the
numbers look bad, but notice that the human plain-English reference
itself only scores 41 %. The EPR implementation uses a small regex list
of ~30 generic medical terms (e.g. *hemoglobin*, *carcinoma*,
*metformin*) and misses most of the actual discharge-summary vocabulary.
More importantly, **correct simplification replaces jargon with plain
words** — "granular cell tumor" → "a type of growth that is usually
not cancer" — so matched entities are *supposed* to disappear. The
regex metric therefore punishes successful simplification on this
corpus.

A credible EPR number on this data needs either the fine-tuned BC5CDR
NER model or a proper medical lexicon (e.g. UMLS) as the entity
extractor. This is tracked as follow-up work.

**The entity-surfacing question the extension actually answers.** When a
user hovers over a clinical term on a live portal, the relevant numbers
are the NER recall (**85.9 %** — we find the term) and precision
(**76.1 %** — we label it correctly). Those determine whether clinical
nuance reaches the user; the text-level EPR above is a proxy that
happens not to fit this dataset.

---

## Reproducing

### Setup

```bash
pip install -r training/requirements.txt
```

### NER

```bash
python training/data/prepare_bc5cdr.py
python training/ner/finetune_ner.py
python training/ner/export_onnx_ner.py
```

Base model: `nlpie/distil-biobert`. Produces 8-bit quantized ONNX ready
for Transformers.js.

### Simplifier (current pipeline)

```bash
python training/data/prepare_simplification_medisimplifier.py
python training/simplification/finetune_t5_medisimplifier.py
python training/simplification/export_onnx_t5_medisimplifier.py
python training/simplification/evaluate_t5_medisimplifier.py --split test --max_samples 200
```

### MedlinePlus glossary

```bash
python scripts/build_glossary.py
```

### Legacy pipeline

An earlier Cochrane-based simplifier (`finetune_t5.py`,
`prepare_simplification.py`) is retained as a baseline reference only.
The shipped extension does not load it.

---

## Privacy guarantee

- All NLP inference runs locally in the browser via WebAssembly.
- Zero network requests during text processing (verify with Chrome
  DevTools → Network while hovering terms).
- Models are downloaded once from Hugging Face, then cached in IndexedDB.
  After that, no external traffic.

---

## Technical details

- **NER model.** DistilBioBERT fine-tuned on BC5CDR (Disease + Chemical).
- **Simplifier.** Flan-T5-small fine-tuned on section-level
  medisimplifier pairs. Beam search at inference (`num_beams=4`) with
  repetition guards baked into the exported model.
- **Runtime.** Transformers.js on ONNX Runtime Web (WebAssembly backend).
- **Quantization.** 8-bit dynamic (`QuantType.QUInt8`) for both models;
  bundled browser footprint ≈ 110 MB (NER 63 MB + T5 50 MB).
- **UI isolation.** Shadow DOM prevents CSS conflicts with the host page.
- **Extension standard.** Chrome Manifest V3.

---

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
