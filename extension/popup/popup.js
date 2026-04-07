const toggleActive = document.getElementById('toggle-active');
const thresholdSlider = document.getElementById('confidence-threshold');
const thresholdValue = document.getElementById('threshold-value');
const nerStatus = document.getElementById('ner-status');
const t5Status = document.getElementById('t5-status');
const entitiesFound = document.getElementById('entities-found');
const simplificationsMade = document.getElementById('simplifications-made');
const glossaryLookups = document.getElementById('glossary-lookups');

function setStatusBadge(el, status) {
  el.className = 'status-badge';
  switch (status) {
    case 'loaded':
      el.textContent = 'Loaded';
      el.classList.add('loaded');
      break;
    case 'loading':
      el.textContent = 'Loading...';
      el.classList.add('loading');
      break;
    case 'error':
      el.textContent = 'Error';
      el.classList.add('error');
      break;
    default:
      el.textContent = 'Not Loaded';
      break;
  }
}

chrome.storage.sync.get(
  { active: true, confidenceThreshold: 0.75 },
  (settings) => {
    toggleActive.checked = settings.active;
    thresholdSlider.value = settings.confidenceThreshold;
    thresholdValue.textContent = settings.confidenceThreshold.toFixed(2);
  }
);

toggleActive.addEventListener('change', () => {
  const active = toggleActive.checked;
  chrome.storage.sync.set({ active });
  chrome.runtime.sendMessage({ type: 'TOGGLE_ACTIVE', active });
});

thresholdSlider.addEventListener('input', () => {
  const val = parseFloat(thresholdSlider.value);
  thresholdValue.textContent = val.toFixed(2);
  chrome.storage.sync.set({ confidenceThreshold: val });
  chrome.runtime.sendMessage({ type: 'UPDATE_THRESHOLD', confidenceThreshold: val });
});

function refreshStats() {
  chrome.runtime.sendMessage({ type: 'GET_STATUS' }, (response) => {
    if (!response) return;
    setStatusBadge(nerStatus, response.nerStatus || 'not_loaded');
    setStatusBadge(t5Status, response.t5Status || 'not_loaded');
    entitiesFound.textContent = response.entitiesFound || 0;
    simplificationsMade.textContent = response.simplificationsMade || 0;
    glossaryLookups.textContent = response.glossaryLookups || 0;
  });
}

refreshStats();
setInterval(refreshStats, 2000);
