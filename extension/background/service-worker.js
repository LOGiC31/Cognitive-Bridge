const state = {
  active: true,
  confidenceThreshold: 0.75,
  nerStatus: 'not_loaded',
  t5Status: 'not_loaded',
  entitiesFound: 0,
  simplificationsMade: 0,
  glossaryLookups: 0,
};

chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    chrome.storage.sync.set({
      active: true,
      confidenceThreshold: 0.75,
    });
    console.log('[CognitiveBridge] Extension installed.');
  } else if (details.reason === 'update') {
    console.log('[CognitiveBridge] Extension updated to version', chrome.runtime.getManifest().version);
  }
});

chrome.runtime.onStartup.addListener(() => {
  chrome.storage.sync.get({ active: true, confidenceThreshold: 0.75 }, (settings) => {
    state.active = settings.active;
    state.confidenceThreshold = settings.confidenceThreshold;
  });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message.type) {
    case 'GET_STATUS':
      sendResponse({ ...state });
      return true;

    case 'TOGGLE_ACTIVE':
      state.active = message.active;
      broadcastToContentScripts({ type: 'TOGGLE_ACTIVE', active: state.active });
      break;

    case 'UPDATE_THRESHOLD':
      state.confidenceThreshold = message.confidenceThreshold;
      broadcastToContentScripts({
        type: 'UPDATE_THRESHOLD',
        confidenceThreshold: state.confidenceThreshold,
      });
      break;

    case 'MODEL_STATUS':
      if (message.model === 'ner') {
        state.nerStatus = message.status;
      } else if (message.model === 't5') {
        state.t5Status = message.status;
      }
      break;

    case 'UPDATE_STATS':
      state.entitiesFound += message.entitiesFound || 0;
      state.simplificationsMade += message.simplificationsMade || 0;
      state.glossaryLookups += message.glossaryLookups || 0;
      break;

    default:
      break;
  }

  sendResponse({ ok: true });
  return true;
});

function broadcastToContentScripts(message) {
  chrome.tabs.query({}, (tabs) => {
    for (const tab of tabs) {
      if (tab.id) {
        chrome.tabs.sendMessage(tab.id, message).catch(() => {});
      }
    }
  });
}
