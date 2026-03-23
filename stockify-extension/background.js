importScripts('config.js');

const REFRESH_INTERVAL_MIN = 300; // refresh every 60 minutes

async function runPrediction(symbol) {
  const res = await fetch(`${CONFIG.API_URL}/predict?symbol=${encodeURIComponent(symbol)}`);
  if (!res.ok) {
    throw new Error(`Prediction failed with status ${res.status}`);
  }

  return res.json();
}

// Schedule alarm
chrome.alarms.create("refreshStocks", { periodInMinutes: REFRESH_INTERVAL_MIN });

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name !== "refreshStocks") return;

  chrome.storage.local.get({stocks: []}, async (res) => {
    for (let symbol of res.stocks) {
      try {
        await fetch(`${CONFIG.API_URL}/predict?symbol=${symbol}`);
      } catch (err) {
        console.error(`Failed to refresh ${symbol}:`, err);
      }
    }
  });
});

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
