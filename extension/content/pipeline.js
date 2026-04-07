import { pipeline, env } from '@xenova/transformers';

env.allowLocalModels = false;
env.useBrowserCache = true;

const NER_MODEL = 'Xenova/bert-base-NER';
const T5_MODEL = 'Xenova/LaMini-Flan-T5-77M';

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
      this.nerPipeline = await pipeline('token-classification', NER_MODEL, {
        quantized: true,
      });
      this.reportModelStatus('ner', 'loaded');
      return this.nerPipeline;
    } catch (err) {
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
      this.t5Pipeline = await pipeline('text2text-generation', T5_MODEL, {
        quantized: true,
      });
      this.reportModelStatus('t5', 'loaded');
      return this.t5Pipeline;
    } catch (err) {
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

    if (entities.length === 0) return [];

    const results = [];

    for (const entity of entities) {
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
