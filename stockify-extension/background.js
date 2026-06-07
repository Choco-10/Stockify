importScripts('config.js');

async function runPrediction(symbol) {
  const apiUrl = await getConfiguredApiUrl();
  const res = await fetch(`${apiUrl}/predict?symbol=${encodeURIComponent(symbol)}`);
  if (!res.ok) {
    throw new Error(`Prediction failed with status ${res.status}`);
  }

  return res.json();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || message.type !== 'runPrediction') {
    return;
  }

  const symbol = (message.symbol || '').toUpperCase().trim();
  if (!symbol) {
    sendResponse({ ok: false, error: 'Missing symbol' });
    return;
  }

  chrome.storage.local.set({
    analysisState: {
      status: 'running',
      symbol,
      hasModel: message.hasModel || false,
      startedAt: Date.now()
    }
  });

  runPrediction(symbol)
    .then((data) => {
      chrome.storage.local.set({
        analysisState: {
          status: 'done',
          symbol,
          finishedAt: Date.now(),
          data
        }
      });
      sendResponse({ ok: true, data });
    })
    .catch((error) => {
      chrome.storage.local.set({
        analysisState: {
          status: 'error',
          symbol,
          finishedAt: Date.now(),
          error: error.message || 'Unknown error'
        }
      });
      sendResponse({ ok: false, error: error.message || 'Unknown error' });
    });

  return true;
});
