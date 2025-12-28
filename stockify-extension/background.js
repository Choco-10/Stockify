const API_URL = "https://stockify-server-90a0.onrender.com";
const REFRESH_INTERVAL_MIN = 300; // refresh every 60 minutes

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
