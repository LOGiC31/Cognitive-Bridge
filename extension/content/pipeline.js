import { pipeline, env } from '@xenova/transformers';

env.allowLocalModels = false;
env.useBrowserCache = true;

const NER_MODEL = 'Xenova/bert-base-NER';
const T5_MODEL = 'Xenova/LaMini-Flan-T5-77M';

// Custom fine-tuned models (swap in once training quality improves):
// const NER_MODEL = 'LOGiC31/cognitive-bridge-ner';
// const T5_MODEL = 'LOGiC31/cognitive-bridge-t5';

const SIMPLIFY_MAX_LENGTH = 256;
const SIMPLIFY_MIN_LENGTH = 10;

export class MedicalPipeline {
  constructor() {
    this.nerPipeline = null;
    this.t5Pipeline = null;
    this.glossary = null;
    this.confidenceThreshold = 0.75;
    this.nerLoading = false;
    this.t5Loading = false;
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
      this.reportModelStatus('t5', 'error');
      throw err;
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

    for (const entity of entities) {
      const avgScore = entity.score;

      if (avgScore >= this.confidenceThreshold) {
        console.log(`[CognitiveBridge] "${entity.word}" (${avgScore.toFixed(3)}) -> T5 simplification`);
        const simplified = await this.simplifyEntity(text, entity);
        if (simplified) {
          console.log(`[CognitiveBridge] T5 output: "${simplified.slice(0, 100)}"`);
        }
        results.push({
          ...entity,
          type: simplified ? 'simplification' : 'glossary',
          explanation: simplified || this.glossaryLookup(entity.word, glossary),
        });
      } else {
        console.log(`[CognitiveBridge] "${entity.word}" (${avgScore.toFixed(3)}) -> glossary fallback`);
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

      const sentence = extractSentence(fullText, entity.start, entity.end);
      const prompt = `Simplify this medical text for a patient: ${sentence}`;

      const output = await t5(prompt, {
        max_new_tokens: SIMPLIFY_MAX_LENGTH,
        min_length: SIMPLIFY_MIN_LENGTH,
        do_sample: false,
      });

      const simplified = output[0]?.generated_text?.trim();
      if (simplified && simplified.length > 5 && simplified !== sentence) {
        return simplified;
      }
      return null;
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

function extractSentence(text, start, end) {
  const before = text.lastIndexOf('.', start);
  const after = text.indexOf('.', end);
  const sentStart = before === -1 ? 0 : before + 1;
  const sentEnd = after === -1 ? text.length : after + 1;
  return text.slice(sentStart, sentEnd).trim();
}

function cleanToken(word, isContinuation = false) {
  const cleaned = word.replace(/^##/, '');
  return isContinuation ? cleaned : ` ${cleaned}`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
