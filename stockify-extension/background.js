const API_URL = "http://127.0.0.1:8000";
const REFRESH_INTERVAL_MIN = 60; // refresh every 60 minutes

// Schedule alarm
chrome.alarms.create("refreshStocks", { periodInMinutes: REFRESH_INTERVAL_MIN });

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name !== "refreshStocks") return;

  chrome.storage.local.get({stocks: []}, async (res) => {
    for (let symbol of res.stocks) {
      try {
        await fetch(`${API_URL}/predict?symbol=${symbol}`);
      } catch (err) {
        console.error(`Failed to refresh ${symbol}:`, err);
      }
    }
  });
});
