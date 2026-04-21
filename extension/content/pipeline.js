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
    // Supplement NER with regex-derived entities for phrases the BC5CDR-style model
    // won't label (e.g., metastatic processes / lymph node involvement).
    for (const extra of extractRegexEntities(text)) {
      entities.push(extra);
    }

    console.log(`[CognitiveBridge] NER found ${entities.length} entities in: "${text.slice(0, 80)}..."`);
    if (entities.length > 0) {
      console.table(entities.map(e => ({ word: e.word, label: e.label, score: e.score.toFixed(3) })));
    }

    if (entities.length === 0) return [];

    const results = [];
    const seen = new Set();

    const GENERIC_FALLBACK_PREFIX = 'Medical term: "';

    for (const entity of entities) {
      const termKey = normalizeKey(entity.word);
      if (isLowValueTerm(termKey)) {
        console.log(`[CognitiveBridge] SKIP low-value term "${entity.word.trim()}"`);
        continue;
      }

      // Skip partial tokens — NER sometimes truncates mid-word (e.g. "lower extre"
      // from "lower extremity"). Check the word appears as a complete token in source.
      const wordTrimmed = entity.word.trim();
      const partialRe = new RegExp(`(?<![a-zA-Z])${escapeRegExp(wordTrimmed)}(?![a-zA-Z])`, 'i');
      if (!partialRe.test(text)) {
        console.log(`[CognitiveBridge] SKIP partial token "${wordTrimmed}" (not a complete word in source)`);
        continue;
      }

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
        const explanation =
          simplified ||
          this.glossaryLookup(entity.word, glossary) ||
          safeBuiltinDefinition(entity.word);
        const type = simplified ? 'simplification' : 'glossary';
        if (!simplified) {
          console.log(`[CognitiveBridge] T5 FALLBACK→GLOSSARY "${entity.word}" (T5 returned null)`);
        }
        // Don't highlight if only explanation is the generic fallback — no value to user.
        if (!explanation || explanation.startsWith(GENERIC_FALLBACK_PREFIX)) {
          console.log(`[CognitiveBridge] SKIP no-explanation "${entity.word}" (generic fallback only)`);
          continue;
        }
        results.push({ ...entity, type, explanation });
      } else {
        console.log(`[CognitiveBridge] LOW CONF→GLOSSARY "${entity.word}" (score: ${avgScore.toFixed(3)} < threshold ${this.confidenceThreshold})`);
        const definition =
          this.glossaryLookup(entity.word, glossary) ||
          safeBuiltinDefinition(entity.word);
        // Don't highlight low-confidence terms with no glossary match either.
        if (!definition || definition.startsWith(GENERIC_FALLBACK_PREFIX)) {
          console.log(`[CognitiveBridge] SKIP low-conf+no-glossary "${entity.word}"`);
          continue;
        }
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
        const isSubword = token.word.startsWith('##');
        current.word += cleanToken(token.word, isSubword);
        current.score = (current.score * current.tokenCount + token.score) / (current.tokenCount + 1);
        current.end = token.end;
        current.tokenCount += 1;
      } else {
        if (current) merged.push(current);
        current = null;
      }
    }

    if (current) merged.push(current);
    return merged
      .map(e => ({ ...e, word: e.word.replace(/[\s,\.;:!?\-—–]+$/, '').trimStart() }))
      .filter((e) => e.word.length > 1);
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
        !isLowQualitySimplification(term, entity.label, simplified) &&
        !isTautologicalSimplification(term, simplified)
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
      console.log(`[CognitiveBridge] GLOSSARY exact "${term}" → "${glossary[normalized].definition.slice(0, 60)}…"`);
      return glossary[normalized].definition;
    }

    const stemmed = normalized.replace(/(ies)$/, 'y')
      .replace(/(es|s)$/, '')
      .replace(/(ing)$/, '')
      .replace(/(ed)$/, '');

    if (glossary[stemmed]) {
      console.log(`[CognitiveBridge] GLOSSARY stem "${term}" (→"${stemmed}") → "${glossary[stemmed].definition.slice(0, 60)}…"`);
      return glossary[stemmed].definition;
    }

    for (const key of Object.keys(glossary)) {
      if (key.includes(normalized) || normalized.includes(key)) {
        console.log(`[CognitiveBridge] GLOSSARY substring "${term}" (matched key "${key}") → "${glossary[key].definition.slice(0, 60)}…"`);
        return glossary[key].definition;
      }
    }

    console.log(`[CognitiveBridge] GLOSSARY miss "${term}"`);
    return null;
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

function buildSimplificationPrompt(fullText, entity) {
  const term = entity.word.trim().replace(/[,\.;:!?]+$/, '');
  // Single-word symptoms (e.g. "dyspnea", "edema") don't benefit from sentence context.
  // Context can cause "entity contamination" where the model explains a more prominent
  // diagnosis in the sentence instead of the target term.
  if (!/\s/.test(term)) {
    return SIMPLIFY_PREFIX + term;
  }
  // Use surrounding sentence so T5 sees full clinical context (e.g. "Malignant
  // neoplasm of the lung with metastatic infiltration" instead of bare "neoplasm").
  // Fall back to term-only when no meaningful sentence window is found.
  const sentence = extractSentence(fullText, entity.word, entity.start, entity.end);
  if (sentence && sentence.length > term.length + 15) {
    // Instruction/action sentences make T5 paraphrase the action instead of defining
    // the term (e.g. "Continue metformin..." → "Keep taking metformin"). Fall back to
    // term-only so T5 uses its medical knowledge rather than echoing the instruction.
    const isInstruction = /\b(continue|initiate|monitor|reassess|start|take|inject|administer|refer|order|recheck|obtain|place[d]?|ordered|noted)\b/i.test(sentence);
    if (!isInstruction) {
      // Prepend term so each entity gets a unique prompt even when multiple
      // terms share the same extracted sentence — prevents cache collisions.
      return SIMPLIFY_PREFIX + term + ': ' + sentence;
    }
  }
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
  // Patient-identifying-ish demo artifacts: avoid caching outputs that echo names.
  // (Also helps block "Doe is a 68-year-old..." templated completions.)
  if (/\b(ms\.|mr\.|mrs\.)\s+[a-z]+\b/i.test(text.trim())) return false;
  if (/\bdoe\b/i.test(text)) return false;
  return true;
}

/** Drop repetitive template completions the model sometimes emits off-distribution. */
function isLowQualitySimplification(term, label, text) {
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
  // Instruction paraphrase — T5 echoed a clinical action sentence instead of defining the term
  if (/^(keep taking|continue taking|take one|inject|administer|initiate|monitor|reassess|recheck)\b/i.test(text.trim())) return true;
  // Patient-intro templating rather than term definition.
  if (/^(the patient|patient|ms\.|mr\.|mrs\.)\s+[a-z]+.*\b(year[-\s]?old|female|male)\b/i.test(text.trim())) return true;
  // Name-based patient intro (e.g. "Doe is a 68-year-old...") — demo artifact / non-definition.
  if (/^[A-Z][a-z]+\s+is\s+a\s+\d{1,3}-year-old\b/.test(text.trim())) return true;
  if (/\bdoe\b/i.test(text)) return true;

  // Block scary / clearly-wrong hallucinations that show up in the logs.
  const termNorm = normalizeKey(term);
  const isAcronym = /^[A-Z]{2,6}$/.test(term.trim());
  const isChemical = (label || '').toLowerCase() === 'chemical';
  if (/(type of (blood )?cancer|blood cancer|leukemia|lymphoma|carcinoma|tumou?r|malignan|metast)/i.test(text)) {
    // If the term itself isn't a cancer-ish term, don't allow cancer claims.
    if (!/(cancer|carcinoma|sarcoma|lymphoma|leukemia|tumou?r|malignan|metast)/i.test(termNorm)) return true;
    // Drugs being described as cancers is a common failure mode.
    if (isChemical) return true;
  }
  // Surgery/procedure hallucinations for lab tests/acronyms (e.g. TSH → surgery).
  if (isAcronym && /(surgery|procedure|operation|biopsy)/i.test(text)) return true;
  // "X (a ...)" pattern is often wrong for our terms (see logs for drugs/diseases).
  if (new RegExp(`^${escapeRegExp(term.trim())}\\s*\\(`).test(text.trim())) return true;
  return false;
}

function isTautologicalSimplification(term, text) {
  // Catches "Heart failure causing severe heart failure" — output restates the term
  // without adding meaning. Only flag when output is short (no new info added).
  const termNorm = term.toLowerCase().replace(/[^a-z]/g, '');
  const textNorm = text.toLowerCase().replace(/[^a-z]/g, '');
  if (text.length < 80 && termNorm.length > 4 && textNorm.includes(termNorm)) {
    // Allow if output clearly adds a plain-language synonym (e.g. "called X" pattern)
    if (!/\b(called|known as|refers to|means|is a|is when|is the)\b/i.test(text)) return true;
  }
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

function normalizeKey(term) {
  return term.toLowerCase().replace(/[^a-z0-9\s]/g, '').replace(/\s+/g, ' ').trim();
}

function isLowValueTerm(normalizedKey) {
  // Words that frequently get tagged but are not helpful to explain.
  return new Set(['intermittent', 'exertion']).has(normalizedKey);
}

function safeBuiltinDefinition(term) {
  const key = normalizeKey(term);
  const map = {
    headache: 'Headache means pain in your head.',
    nausea: 'Nausea means feeling like you might vomit.',
    wheezing: 'Wheezing means a whistling sound when you breathe, often from narrowed airways.',
    migraine: 'A migraine is a type of severe headache that can come with nausea and sensitivity to light or sound.',
    photophobia: 'Photophobia means sensitivity to light.',
    albuterol: 'Albuterol is an inhaler medicine that helps open the airways to make breathing easier.',
    tsh: 'TSH is a blood test that helps check how well your thyroid is working.',
  };
  return map[key] || null;
}

function extractRegexEntities(text) {
  const results = [];
  const seen = new Set();

  // Captures common staging/oncology phrasing that the NER model likely misses.
  const patterns = [
    /\bmetastatic infiltration to (?:the )?(?:[a-z-]+\s+){0,4}lymph nodes\b/ig,
    /\bmediastinal lymph nodes\b/ig,
  ];

  for (const re of patterns) {
    re.lastIndex = 0;
    let match;
    while ((match = re.exec(text)) !== null) {
      const word = match[0].trim();
      const key = word.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      results.push({
        word,
        label: 'Disease',
        score: 1.0,
        start: match.index,
        end: match.index + word.length,
        tokenCount: Math.max(1, word.split(/\s+/).length),
      });
    }
  }

  return results;
}
