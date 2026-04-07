const HIGHLIGHT_STYLES = `
  .cb-highlight {
    border-radius: 2px;
    padding: 0 2px;
    cursor: pointer;
    position: relative;
    transition: background-color 0.2s;
  }
  .cb-highlight[data-label="Disease"],
  .cb-highlight[data-label="MISC"],
  .cb-highlight[data-label="disease"] {
    background-color: rgba(254, 202, 202, 0.6);
    border-bottom: 2px solid #ef4444;
  }
  .cb-highlight[data-label="Disease"]:hover,
  .cb-highlight[data-label="disease"]:hover {
    background-color: rgba(254, 202, 202, 0.9);
  }
  .cb-highlight[data-label="Chemical"],
  .cb-highlight[data-label="chemical"] {
    background-color: rgba(191, 219, 254, 0.6);
    border-bottom: 2px solid #3b82f6;
  }
  .cb-highlight[data-label="Chemical"]:hover,
  .cb-highlight[data-label="chemical"]:hover {
    background-color: rgba(191, 219, 254, 0.9);
  }
  .cb-highlight[data-label="PER"],
  .cb-highlight[data-label="LOC"],
  .cb-highlight[data-label="ORG"] {
    background-color: rgba(217, 249, 157, 0.6);
    border-bottom: 2px solid #84cc16;
  }
  .cb-tooltip-container {
    position: fixed;
    z-index: 2147483647;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.2s ease;
  }
  .cb-tooltip-container.visible {
    opacity: 1;
    pointer-events: auto;
  }
  .cb-tooltip {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.12), 0 4px 10px rgba(0,0,0,0.06);
    padding: 14px 16px;
    max-width: 400px;
    min-width: 200px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 14px;
    line-height: 1.5;
    color: #334155;
  }
  .cb-tooltip-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid #f1f5f9;
  }
  .cb-tooltip-term {
    font-weight: 700;
    font-size: 15px;
    color: #0f172a;
  }
  .cb-tooltip-badge {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 2px 8px;
    border-radius: 10px;
  }
  .cb-tooltip-badge.disease {
    background: #fee2e2;
    color: #991b1b;
  }
  .cb-tooltip-badge.chemical {
    background: #dbeafe;
    color: #1e40af;
  }
  .cb-tooltip-badge.other {
    background: #f0fdf4;
    color: #166534;
  }
  .cb-tooltip-badge.simplification {
    background: #f0fdf4;
    color: #166534;
  }
  .cb-tooltip-badge.glossary {
    background: #fef3c7;
    color: #92400e;
  }
  .cb-tooltip-body {
    color: #475569;
    font-size: 13px;
  }
  .cb-tooltip-confidence {
    margin-top: 8px;
    font-size: 11px;
    color: #94a3b8;
  }
  .cb-tooltip-source {
    margin-top: 4px;
    font-size: 11px;
    color: #94a3b8;
    font-style: italic;
  }
`;

export class TooltipManager {
  constructor() {
    this.shadowHost = null;
    this.shadowRoot = null;
    this.tooltipContainer = null;
    this.activeHighlight = null;
    this.setupShadowDOM();
  }

  setupShadowDOM() {
    this.shadowHost = document.createElement('div');
    this.shadowHost.id = 'cognitive-bridge-tooltips';
    document.body.appendChild(this.shadowHost);
    this.shadowRoot = this.shadowHost.attachShadow({ mode: 'closed' });

    const style = document.createElement('style');
    style.textContent = HIGHLIGHT_STYLES;
    this.shadowRoot.appendChild(style);

    this.tooltipContainer = document.createElement('div');
    this.tooltipContainer.className = 'cb-tooltip-container';
    this.shadowRoot.appendChild(this.tooltipContainer);
  }

  applyHighlights(node, entities) {
    if (!node || !entities || entities.length === 0) return;

    const text = node.textContent || '';

    const sorted = [...entities].sort((a, b) => (b.start ?? 0) - (a.start ?? 0));

    for (const entity of sorted) {
      const entityText = entity.word.trim();
      const startIdx = text.toLowerCase().indexOf(entityText.toLowerCase());
      if (startIdx === -1) continue;

      this.highlightRange(node, startIdx, startIdx + entityText.length, entity);
    }

    node.classList.add('cb-processed');
  }

  highlightRange(container, start, end, entity) {
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null, false);
    let offset = 0;
    let node;

    while ((node = walker.nextNode())) {
      const nodeLen = node.textContent.length;
      const nodeStart = offset;
      const nodeEnd = offset + nodeLen;

      if (nodeStart < end && nodeEnd > start) {
        const overlapStart = Math.max(start - nodeStart, 0);
        const overlapEnd = Math.min(end - nodeStart, nodeLen);

        if (overlapStart > 0) {
          node.splitText(overlapStart);
          node = walker.nextNode();
          if (!node) break;
        }

        const remaining = node.textContent.length;
        const splitPos = overlapEnd - (overlapStart > 0 ? overlapStart : 0);
        if (splitPos < remaining) {
          node.splitText(splitPos);
        }

        const mark = document.createElement('mark');
        mark.className = 'cb-highlight';
        mark.dataset.label = entity.label || 'other';
        mark.dataset.entity = JSON.stringify({
          word: entity.word,
          label: entity.label,
          type: entity.type,
          explanation: entity.explanation,
          score: entity.score,
        });

        node.parentNode.insertBefore(mark, node);
        mark.appendChild(node);

        this.attachTooltipListeners(mark);
        break;
      }

      offset += nodeLen;
    }
  }

  attachTooltipListeners(mark) {
    mark.addEventListener('mouseenter', (e) => this.showTooltip(e, mark));
    mark.addEventListener('mouseleave', () => this.hideTooltip());
    mark.addEventListener('click', (e) => {
      e.preventDefault();
      if (this.activeHighlight === mark) {
        this.hideTooltip();
      } else {
        this.showTooltip(e, mark);
      }
    });
  }

  showTooltip(event, mark) {
    this.activeHighlight = mark;

    let entityData;
    try {
      entityData = JSON.parse(mark.dataset.entity);
    } catch {
      return;
    }

    const labelClass = (entityData.label || '').toLowerCase().includes('disease')
      ? 'disease'
      : (entityData.label || '').toLowerCase().includes('chemical')
        ? 'chemical'
        : 'other';

    const typeClass = entityData.type === 'simplification' ? 'simplification' : 'glossary';
    const typeLabel = entityData.type === 'simplification' ? 'AI Simplified' : 'Glossary';

    this.tooltipContainer.innerHTML = `
      <div class="cb-tooltip">
        <div class="cb-tooltip-header">
          <span class="cb-tooltip-term">${escapeHtml(entityData.word)}</span>
          <span class="cb-tooltip-badge ${labelClass}">${escapeHtml(entityData.label || 'Entity')}</span>
          <span class="cb-tooltip-badge ${typeClass}">${typeLabel}</span>
        </div>
        <div class="cb-tooltip-body">
          ${escapeHtml(entityData.explanation || 'No explanation available.')}
        </div>
        <div class="cb-tooltip-confidence">
          Confidence: ${((entityData.score || 0) * 100).toFixed(0)}%
        </div>
        <div class="cb-tooltip-source">
          ${entityData.type === 'glossary' ? 'Source: MedlinePlus' : 'Locally simplified — verify with your provider'}
        </div>
      </div>
    `;

    const rect = mark.getBoundingClientRect();
    const tooltipEl = this.tooltipContainer;

    let top = rect.bottom + 8;
    let left = rect.left;

    tooltipEl.style.position = 'fixed';
    tooltipEl.style.top = `${top}px`;
    tooltipEl.style.left = `${left}px`;
    tooltipEl.classList.add('visible');

    requestAnimationFrame(() => {
      const tipRect = tooltipEl.getBoundingClientRect();
      if (tipRect.right > window.innerWidth - 10) {
        tooltipEl.style.left = `${window.innerWidth - tipRect.width - 10}px`;
      }
      if (tipRect.bottom > window.innerHeight - 10) {
        tooltipEl.style.top = `${rect.top - tipRect.height - 8}px`;
      }
    });
  }

  hideTooltip() {
    this.activeHighlight = null;
    this.tooltipContainer.classList.remove('visible');
  }

  removeAllHighlights() {
    const highlights = document.querySelectorAll('.cb-highlight');
    for (const mark of highlights) {
      const parent = mark.parentNode;
      while (mark.firstChild) {
        parent.insertBefore(mark.firstChild, mark);
      }
      parent.removeChild(mark);
      parent.normalize();
    }
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
