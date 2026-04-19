import { pipeline, env } from '@xenova/transformers';

env.allowLocalModels = false;
env.useBrowserCache = true;

// Match training/data/prepare_simplification.py PREFIX (full sentence after prefix, not "term means").
try {
  const ortEnv = env.backends?.onnx?.env;
  if (ortEnv) ortEnv.logLevel = 'error';
} catch {
  // ignore if onnx env shape differs across versions
}

const NER_MODEL = 'LOGiC31/cognitive-bridge-ner';
const T5_MODEL = 'LOGiC31/cognitive-bridge-t5-medisimplifier';

/**
 * We show the output as a tooltip "term explanation", so the generation prompt must be
 * term-focused (not a full-sentence rewrite).
 *
 * Keep the original training prefix, but add an explicit instruction to explain the term
 * using the sentence as context.
 */
const SIMPLIFY_PREFIX = 'Simplify this medical text for a patient: ';

const SIMPLIFY_MAX_NEW_TOKENS = 72;

export class MedicalPipeline {
  constructor() {
    this.nerPipeline = null;
    this.t5Pipeline = null;
    this.glossary = null;
    this.confidenceThreshold = 0.75;
    this.nerLoading = false;
    this.t5Loading = false;
    this.t5Failed = false;
    /** Keyed by full simplification prompt (context + term), not bare term. */
    this.sentenceCache = new Map();
    /** Keyed by normalized term — best result seen across all DOM blocks. */
    this.termResultCache = new Map();
  }

  setThreshold(val) {
    this.confidenceThreshold = val;
  }

  async loadNER() {
    if (this.nerPipeline) return this.nerPipeline;
    if (this.nerLoading) {
      while (this.nerLoading) await sleep(100);
      return this.nerPipeline;
    }

    this.nerLoading = true;
    this.reportModelStatus('ner', 'loading');

    try {
      console.log(`[CognitiveBridge] Loading NER model: ${NER_MODEL} (quantized: true)`);
      this.nerPipeline = await pipeline('token-classification', NER_MODEL, {
        quantized: true,
      });
      console.log(`[CognitiveBridge] NER model loaded successfully: ${NER_MODEL}`);
      this.reportModelStatus('ner', 'loaded');
      return this.nerPipeline;
    } catch (err) {
      console.error(`[CognitiveBridge] NER model failed to load:`, err);
      this.reportModelStatus('ner', 'error');
      throw err;
    } finally {
      this.nerLoading = false;
    }
  }

  async loadT5() {
    if (this.t5Pipeline) return this.t5Pipeline;
    if (this.t5Failed) return null;
    if (this.t5Loading) {
      while (this.t5Loading) await sleep(100);
      return this.t5Pipeline;
    }

    this.t5Loading = true;
    this.reportModelStatus('t5', 'loading');

    try {
      console.log(`[CognitiveBridge] Loading T5 model: ${T5_MODEL} (quantized: true)`);
      this.t5Pipeline = await pipeline('text2text-generation', T5_MODEL, {
        quantized: true,
      });
      console.log(`[CognitiveBridge] T5 model loaded successfully: ${T5_MODEL}`);
      this.reportModelStatus('t5', 'loaded');
      return this.t5Pipeline;
    } catch (err) {
      console.error(`[CognitiveBridge] T5 model failed to load:`, err);
      this.t5Failed = true;
      this.reportModelStatus('t5', 'error');
      return null;
    } finally {
      this.t5Loading = false;
    }
  }

  async loadGlossary() {
    if (this.glossary) return this.glossary;

    try {
      const url = chrome.runtime.getURL('data/medlineplus_glossary.json');
      const resp = await fetch(url);
      this.glossary = await resp.json();
    } catch {
      this.glossary = {};
    }
    return this.glossary;
  }

  async process(text) {
    const ner = await this.loadNER();
    const glossary = await this.loadGlossary();

    const nerResults = await ner(text, { ignore_labels: [] });
    const entities = this.mergeEntities(nerResults);

    console.log(`[CognitiveBridge] NER found ${entities.length} entities in: "${text.slice(0, 80)}..."`);
    if (entities.length > 0) {
      console.table(entities.map(e => ({ word: e.word, label: e.label, score: e.score.toFixed(3) })));
    }

    if (entities.length === 0) return [];

    const results = [];
    const seen = new Set();

    for (const entity of entities) {
      const termKey = entity.word.trim().toLowerCase();
      if (seen.has(termKey)) {
        const prev = results.find(r => r.word.trim().toLowerCase() === termKey);
        if (prev) {
          results.push({ ...entity, type: prev.type, explanation: prev.explanation });
        }
        continue;
      }
      seen.add(termKey);

      const avgScore = entity.score;

      if (avgScore >= this.confidenceThreshold) {
        const simplified = await this.simplifyEntity(text, entity);
        results.push({
          ...entity,
          type: simplified ? 'simplification' : 'glossary',
          explanation: simplified || this.glossaryLookup(entity.word, glossary),
        });
      } else {
        const definition = this.glossaryLookup(entity.word, glossary);
        results.push({
          ...entity,
          type: 'glossary',
          explanation: definition,
        });
      }
    }

    return results;
  }

  mergeEntities(nerResults) {
    const merged = [];
    let current = null;

    for (const token of nerResults) {
      const bioTag = token.entity || '';
      const isBegin = bioTag.startsWith('B-');
      const isInside = bioTag.startsWith('I-');
      const label = bioTag.replace(/^[BI]-/, '');

      if (isBegin) {
        if (current) merged.push(current);
        current = {
          word: cleanToken(token.word),
          label,
          score: token.score,
          start: token.start,
          end: token.end,
          tokenCount: 1,
        };
      } else if (isInside && current && current.label === label) {
        current.word += cleanToken(token.word, true);
        current.score = (current.score * current.tokenCount + token.score) / (current.tokenCount + 1);
        current.end = token.end;
        current.tokenCount += 1;
      } else {
        if (current) merged.push(current);
        current = null;
      }
    }

    if (current) merged.push(current);
    return merged.filter((e) => e.word.length > 1);
  }

  async simplifyEntity(fullText, entity) {
    try {
      const t5 = await this.loadT5();
      if (!t5) return null;

      const term = entity.word.trim();
      const termKey = term.toLowerCase().replace(/[^a-z0-9]/g, '');

      // Return a previously cached good result for this term.
      // If the cached result is bad (e.g. from demo/instructions page), fall through
      // and let this block attempt a better simplification.
      const cached = this.termResultCache.get(termKey);
      if (cached !== undefined && isGoodSimplification(cached)) {
        console.log(`[CognitiveBridge] CACHE HIT "${term}" -> "${cached.slice(0, 60)}…"`);
        return cached;
      }
      if (cached !== undefined) {
        console.log(`[CognitiveBridge] CACHE BAD "${term}" -> retrying with new context`);
      }

      const prompt = buildSimplificationPrompt(fullText, entity);

      if (this.sentenceCache.has(prompt)) {
        console.log(`[CognitiveBridge] PROMPT CACHE HIT "${term}"`);
        return this.sentenceCache.get(prompt);
      }

      const output = await t5(prompt, {
        max_new_tokens: SIMPLIFY_MAX_NEW_TOKENS,
        do_sample: false,
        num_beams: 4,
        early_stopping: true,
        repetition_penalty: 1.15,
        no_repeat_ngram_size: 3,
      });

      const simplifiedRaw = output[0]?.generated_text?.trim();
      console.log(`[CognitiveBridge] RAW OUTPUT "${term}": "${(simplifiedRaw || '').slice(0, 120)}"`);

      const simplified = simplifiedRaw ? cleanModelOutput(term, simplifiedRaw) : simplifiedRaw;
      let result = null;
      if (
        simplified &&
        simplified.length > 5 &&
        simplified.toLowerCase() !== term.toLowerCase() &&
        !isLowQualitySimplification(simplified)
      ) {
        result = simplified;
      }

      this.sentenceCache.set(prompt, result);
      if (result) {
        const good = isGoodSimplification(result);
        console.log(`[CognitiveBridge] T5 ${good ? '✓' : '✗'} "${entity.word}" -> "${result.slice(0, 80)}${result.length > 80 ? '…' : ''}"`);
        if (good) {
          this.termResultCache.set(termKey, result);
        }
      } else {
        console.log(`[CognitiveBridge] FILTERED OUT "${term}" (raw: "${(simplifiedRaw || '').slice(0, 60)}")`);
      }
      return result;
    } catch (err) {
      console.warn('[CognitiveBridge] T5 simplification error:', err);
      return null;
    }
  }

  glossaryLookup(term, glossary) {
    const normalized = term.toLowerCase().replace(/[^a-z0-9\s]/g, '').trim();

    if (glossary[normalized]) {
      return glossary[normalized].definition;
    }

    const stemmed = normalized.replace(/(ies)$/, 'y')
      .replace(/(es|s)$/, '')
      .replace(/(ing)$/, '')
      .replace(/(ed)$/, '');

    if (glossary[stemmed]) {
      return glossary[stemmed].definition;
    }

    for (const key of Object.keys(glossary)) {
      if (key.includes(normalized) || normalized.includes(key)) {
        return glossary[key].definition;
      }
    }

    return `Medical term: "${term}". Ask your healthcare provider for more details.`;
  }

  reportModelStatus(model, status) {
    try {
      chrome.runtime.sendMessage({
        type: 'MODEL_STATUS',
        model,
        status,
      });
    } catch {
      // popup may not be open
    }
  }
}

function buildSimplificationPrompt(_fullText, entity) {
  // Pass only the term — T5 uses its pre-trained medical knowledge to produce
  // a plain-language definition rather than paraphrasing the surrounding context.
  const term = entity.word.trim().replace(/[,\.;:!?]+$/, '');
  return SIMPLIFY_PREFIX + term;
}

/**
 * True if the result looks like a proper medical simplification.
 * Used to gate termResultCache — bad results (from demo/instructions pages or
 * section-header echoes) must not block later blocks from producing a better result.
 */
function isGoodSimplification(text) {
  if (!text || text.length < 10) return false;
  const t = text.toLowerCase();
  // Dev/UI context leaked into output (demo page artifact)
  if (/npm run build|build the extension|open chrome|webpack/i.test(t)) return false;
  // Section header echoed at the start
  if (/^(radiology report|findings|current medications?|laboratory results?|office visit|visit [—–]|assessment|plan\s*[\d:])/i.test(text.trim())) return false;
  // Date-only contamination (e.g. "March 15, 2026 Chief Complaint...")
  if (/^[a-z]+ \d{1,2}, \d{4}/i.test(text.trim())) return false;
  // Visit/admission date summaries with no term-specific content
  if (/^the patient (came to|was admitted to|should see|went to|visited).{0,50}(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}/i.test(text.trim())) return false;
  // Chief Complaint echoed as output
  if (/^chief complaint/i.test(text.trim())) return false;
  return true;
}

/** Drop repetitive template completions the model sometimes emits off-distribution. */
function isLowQualitySimplification(text) {
  const t = text.toLowerCase();
  if ((t.match(/this is a medical text for a patient/g) || []).length >= 2) return true;
  if (/^this is a medical text for a patient\.?\s*$/i.test(text.trim())) return true;
  if (t.includes('update of a previous version')) return true;
  if (t.includes('original version of this review')) return true;
  if (t.includes('medical literature')) return true;
  if (t.startsWith('this is a medical procedure that involves a person\'s blood pressure') && text.length < 160) {
    return true;
  }
  if (/^explain the medical term/i.test(text.trim())) return true;
  return false;
}

function cleanModelOutput(term, text) {
  let t = text.trim();

  // If the model echoed the prompt/question, strip it.
  t = t.replace(/^simplify this medical text for a patient:\s*/i, '');
  t = t.replace(/^in one short sentence,\s*what does\s*"?.+?"?\s*mean for a patient\?\s*/i, '');

  // Strip the exact term prefix some generations start with.
  const termRe = new RegExp(`^("${escapeRegExp(term)}"|${escapeRegExp(term)})\\s*[:-]\\s*`, 'i');
  t = t.replace(termRe, '');

  // Take first sentence for tooltip brevity.
  const idx = t.indexOf('.');
  if (idx !== -1) t = t.slice(0, idx + 1);

  return t.trim();
}

function escapeRegExp(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function extractSentence(text, termWord, _start, _end) {
  // NER character offsets are in tokenizer-normalized (whitespace-collapsed) space,
  // not raw-text space. Large DOM blocks have headers at position 0 so all entity
  // offsets map near 0, producing the same header-contaminated window for every term.
  // Searching for the term directly in raw text gives the correct position.
  const term = termWord.replace(/\s+/g, ' ').trim();
  const cleanTerm = term.replace(/[,\.;:!?]+$/, '');

  let termPos = text.indexOf(term);
  if (termPos === -1) termPos = text.indexOf(cleanTerm);
  if (termPos === -1) termPos = Math.max(0, _start);

  // Scan back in raw text for nearest sentence/paragraph boundary.
  // Newlines are stronger section boundaries than ". " in clinical notes.
  const before = text.slice(0, termPos);
  const nlIdx = before.lastIndexOf('\n');
  const dotIdx = before.lastIndexOf('. ');
  const boundaryIdx = Math.max(nlIdx, dotIdx);
  const sentStartRaw = boundaryIdx === -1 ? Math.max(0, termPos - 200) : boundaryIdx + 1;

  // Scan forward for nearest sentence end.
  const afterEnd = termPos + Math.max(term.length, cleanTerm.length);
  const dotFwdIdx = text.indexOf('.', afterEnd);
  const nlFwdIdx = text.indexOf('\n', afterEnd);
  const fwdBoundary =
    dotFwdIdx !== -1 && nlFwdIdx !== -1 ? Math.min(dotFwdIdx, nlFwdIdx) :
    dotFwdIdx !== -1 ? dotFwdIdx : nlFwdIdx;
  const sentEndRaw = fwdBoundary === -1 ? Math.min(text.length, termPos + 300) : fwdBoundary + 1;

  const sentence = text.slice(sentStartRaw, sentEndRaw).replace(/\s+/g, ' ').trim();
  return sentence.slice(0, 300);
}

function cleanToken(word, isContinuation = false) {
  const cleaned = word.replace(/^##/, '');
  return isContinuation ? cleaned : ` ${cleaned}`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
