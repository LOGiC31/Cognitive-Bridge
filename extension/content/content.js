import { MedicalPipeline } from './pipeline.js';
import { TooltipManager } from './tooltip.js';

const MEDICAL_PATTERNS = [
  /\b\w+(itis|ectomy|emia|osis|emia|pathy|algia|plasty|scopy|ology|gram|tomy)\b/i,
  /\b(diagnosis|prognosis|etiology|pathology|carcinoma|lymphoma|sarcoma)\b/i,
  /\b(mg|mcg|mL|mmol|IU)\/?(kg|dL|L|day)?\b/,
  /\b(acetaminophen|ibuprofen|metformin|lisinopril|atorvastatin|omeprazole|amoxicillin|azithromycin|hydrochlorothiazide|levothyroxine)\b/i,
  /\b(hypertension|diabetes|mellitus|hyperlipidemia|thrombosis|fibrillation|myocardial|infarction|edema|dyspnea|tachycardia|bradycardia)\b/i,
  /\b(hemoglobin|platelet|leukocyte|erythrocyte|creatinine|bilirubin|troponin|albumin)\b/i,
];

const MIN_TEXT_LENGTH = 20;
const DEBOUNCE_MS = 500;

class CognitiveBridge {
  constructor() {
    this.active = true;
    this.pipeline = new MedicalPipeline();
    this.tooltipManager = new TooltipManager();
    this.processedNodes = new WeakSet();
    this.pendingNodes = [];
    this.debounceTimer = null;
    this.observer = null;
  }

  async init() {
    const settings = await this.loadSettings();
    this.active = settings.active;
    this.pipeline.setThreshold(settings.confidenceThreshold);

    this.setupMessageListener();

    if (this.active) {
      this.startObserving();
      this.processExistingContent();
    }
  }

  loadSettings() {
    return new Promise((resolve) => {
      chrome.storage.sync.get(
        { active: true, confidenceThreshold: 0.75 },
        resolve
      );
    });
  }

  setupMessageListener() {
    chrome.runtime.onMessage.addListener((msg) => {
      if (msg.type === 'TOGGLE_ACTIVE') {
        this.active = msg.active;
        if (this.active) {
          this.startObserving();
          this.processExistingContent();
        } else {
          this.stopObserving();
          this.tooltipManager.removeAllHighlights();
        }
      } else if (msg.type === 'UPDATE_THRESHOLD') {
        this.pipeline.setThreshold(msg.confidenceThreshold);
      }
    });
  }

  startObserving() {
    if (this.observer) return;

    this.observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if (mutation.type === 'childList') {
          for (const node of mutation.addedNodes) {
            this.collectTextNodes(node);
          }
        } else if (mutation.type === 'characterData') {
          const parent = mutation.target.parentElement;
          if (parent && !this.processedNodes.has(parent)) {
            this.scheduleNode(parent);
          }
        }
      }
    });

    this.observer.observe(document.body, {
      childList: true,
      subtree: true,
      characterData: true,
    });
  }

  stopObserving() {
    if (this.observer) {
      this.observer.disconnect();
      this.observer = null;
    }
  }

  collectTextNodes(root) {
    if (!root || root.nodeType === Node.COMMENT_NODE) return;

    if (root.nodeType === Node.TEXT_NODE) {
      const parent = root.parentElement;
      if (parent && !this.processedNodes.has(parent)) {
        this.scheduleNode(parent);
      }
      return;
    }

    if (root.nodeType !== Node.ELEMENT_NODE) return;

    const tag = root.tagName;
    if (['SCRIPT', 'STYLE', 'NOSCRIPT', 'SVG', 'CANVAS'].includes(tag)) return;
    if (root.classList && root.classList.contains('cb-processed')) return;

    for (const child of root.childNodes) {
      this.collectTextNodes(child);
    }
  }

  scheduleNode(node) {
    this.pendingNodes.push(node);
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => this.processPendingNodes(), DEBOUNCE_MS);
  }

  async processPendingNodes() {
    const nodes = [...this.pendingNodes];
    this.pendingNodes = [];

    for (const node of nodes) {
      if (this.processedNodes.has(node)) continue;

      const text = node.textContent?.trim();
      if (!text || text.length < MIN_TEXT_LENGTH) continue;
      if (!this.containsMedicalContent(text)) continue;

      this.processedNodes.add(node);

      try {
        const results = await this.pipeline.process(text);
        if (results && results.length > 0) {
          this.tooltipManager.applyHighlights(node, results);
          this.reportStats(results);
        }
      } catch (err) {
        console.error('[CognitiveBridge] Processing error:', err);
      }
    }
  }

  containsMedicalContent(text) {
    return MEDICAL_PATTERNS.some((pattern) => pattern.test(text));
  }

  processExistingContent() {
    const elements = document.querySelectorAll('p, div, span, td, li, dd, dt, h1, h2, h3, h4, h5, h6');
    for (const el of elements) {
      if (!this.processedNodes.has(el)) {
        const text = el.textContent?.trim();
        if (text && text.length >= MIN_TEXT_LENGTH && this.containsMedicalContent(text)) {
          this.scheduleNode(el);
        }
      }
    }
  }

  reportStats(results) {
    const entityCount = results.length;
    const simplifications = results.filter((r) => r.type === 'simplification').length;
    const lookups = results.filter((r) => r.type === 'glossary').length;

    chrome.runtime.sendMessage({
      type: 'UPDATE_STATS',
      entitiesFound: entityCount,
      simplificationsMade: simplifications,
      glossaryLookups: lookups,
    });
  }
}

const bridge = new CognitiveBridge();
bridge.init().catch((err) => console.error('[CognitiveBridge] Init failed:', err));
