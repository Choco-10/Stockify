const searchInput = document.getElementById("stock-search");
const resultsList = document.getElementById("search-results");
const predictBtn = document.getElementById("predictBtn");
const resultsDiv = document.getElementById("results");
const savedDiv = document.getElementById("savedStocks");

const searchContainer = document.querySelector(".search-container");

let selectedSymbol = "";
let timeout = null;
let etaInterval = null;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getCurrencySymbol(currency) {
  const map = {
    "USD": "$",
    "INR": "₹",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "¥",
    "AUD": "A$",
    "CAD": "C$",
    "CHF": "CHF",
  };
  return map[currency] || currency;
}

function formatSeconds(s) {
  if (s == null) return "";
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}m ${sec}s`;
}

function directionColor(dir) {
  if (dir === "UP") return "#16a34a";
  if (dir === "DOWN") return "#dc2626";
  return "#6b7280";
}

function directionArrow(dir) {
  if (dir === "UP") return "&#9650;";
  if (dir === "DOWN") return "&#9660;";
  return "&#9644;";
}

// ---------------------------------------------------------------------------
// Fetch available stock symbols (for ETA pre-check)
// ---------------------------------------------------------------------------

async function fetchAvailableStocks() {
  try {
    const apiUrl = await getConfiguredApiUrl();
    const res = await fetch(`${apiUrl}/available_stocks`);
    const data = await res.json();
    return data.stocks || [];
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Render prediction result
// ---------------------------------------------------------------------------

function renderPredictionHtml(data, symbolCurrency) {
  const cp = data.current_price;
  const np = data.next_day_prediction;
  const dir = data.direction;
  const mode = data.mode;
  const procTime = data.processing_time_seconds;

  const cpStr = cp != null ? `${symbolCurrency}${cp.toFixed(2)}` : "N/A";
  const npStr = np != null ? `${symbolCurrency}${np.toFixed(2)}` : "N/A";
  const dirColor = directionColor(dir);
  const dirArrow = directionArrow(dir);

  const modeLabel = mode === "training"
    ? '<span style="color:#d97706;font-weight:600;">Trained + Inference</span>'
    : '<span style="color:#2563eb;font-weight:600;">Inference</span>';

  const timeLabel = procTime != null ? ` (${formatSeconds(procTime)})` : "";

  return `
    <p style="margin:0 0 6px;font-size:11px;color:#6b7280;">${modeLabel}${timeLabel}</p>
    <p style="margin:2px 0;"><strong>Symbol:</strong> ${data.symbol}</p>
    <p style="margin:2px 0;"><strong>Current Price:</strong> ${cpStr}</p>
    <p style="margin:2px 0;"><strong>Next Day Prediction:</strong> ${npStr}</p>
    <p style="margin:2px 0;">
      <strong>Direction:</strong>
      <span style="color:${dirColor};font-weight:700;">${dirArrow} ${dir}</span>
    </p>
  `;
}

// ---------------------------------------------------------------------------
// Currency resolution
// ---------------------------------------------------------------------------

async function getSymbolCurrency(symbol) {
  try {
    const apiUrl = await getConfiguredApiUrl();
    const stockRes = await fetch(`${apiUrl}/search?q=${encodeURIComponent(symbol)}`);
    const stockList = await stockRes.json();
    const stockData = stockList.find(s => s.symbol && s.symbol.toUpperCase() === symbol.toUpperCase());
    return getCurrencySymbol(stockData?.currency || "USD");
  } catch (err) {
    console.error("Failed to resolve currency", err);
    return getCurrencySymbol("USD");
  }
}

// ---------------------------------------------------------------------------
// Restore state on popup open
// ---------------------------------------------------------------------------

async function restoreAnalysisState() {
  const state = await new Promise((resolve) => {
    chrome.storage.local.get({ analysisState: null }, (res) => resolve(res.analysisState));
  });

  if (!state) return;

  if (state.status === "running") {
    const etaSeconds = state.hasModel ? 5 : 60;
    const etaLabel = state.hasModel ? "Running inference" : "Training from scratch";
    setBtnLoading(true, `Analyzing ${state.symbol}...`);
    resultsDiv.innerHTML = `
      <p style="color:#6b7280;">Analyzing <strong>${state.symbol}</strong>...</p>
      <p style="color:#9ca3af;font-size:12px;" id="eta-display">ETA: calculating...</p>
      <p style="color:#9ca3af;font-size:12px;font-style:italic;">${etaLabel}.</p>
    `;
    startEtaTimer(etaSeconds);
    if (state.symbol) {
      searchInput.value = state.symbol;
      selectedSymbol = state.symbol;
    }
    return;
  }

  if (state.status === "error") {
    setBtnLoading(false);
    resultsDiv.innerHTML = `<p style="color:#dc2626;">Error: ${escapeHtml(state.error || "Prediction failed")}</p>`;
    if (state.symbol) {
      searchInput.value = state.symbol;
      selectedSymbol = state.symbol;
    }
    return;
  }

  if (state.status === "done" && state.data) {
    setBtnLoading(false);
    const symbolCurrency = await getSymbolCurrency(state.symbol || state.data.symbol);
    resultsDiv.innerHTML = renderPredictionHtml(state.data, symbolCurrency);
    if (state.symbol) {
      searchInput.value = state.symbol;
      selectedSymbol = state.symbol;
    }
  }
}

// ---------------------------------------------------------------------------
// Loading button state
// ---------------------------------------------------------------------------

function setBtnLoading(loading, label) {
  if (loading) {
    predictBtn.disabled = true;
    predictBtn.textContent = label || "Analyzing...";
    predictBtn.style.opacity = "0.7";
    predictBtn.style.cursor = "wait";
  } else {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict";
    predictBtn.style.opacity = "1";
    predictBtn.style.cursor = "pointer";
    stopEtaTimer();
  }
}

// ---------------------------------------------------------------------------
// ETA countdown timer
// ---------------------------------------------------------------------------

function startEtaTimer(totalSeconds) {
  stopEtaTimer();
  const endTime = Date.now() + totalSeconds * 1000;
  // Show initial ETA immediately
  updateEtaDisplay(endTime);
  etaInterval = setInterval(() => {
    updateEtaDisplay(endTime);
  }, 1000);
}

function updateEtaDisplay(endTime) {
  const remaining = Math.max(0, Math.ceil((endTime - Date.now()) / 1000));
  const el = document.getElementById("eta-display");
  if (!el) { stopEtaTimer(); return; }
  if (remaining <= 0) {
    el.textContent = "ETA: almost done...";
    return;
  }
  if (remaining >= 60) {
    const m = Math.floor(remaining / 60);
    const s = remaining % 60;
    el.textContent = `ETA: ~${m}m ${s}s`;
  } else {
    el.textContent = `ETA: ~${remaining}s`;
  }
}

function stopEtaTimer() {
  if (etaInterval) {
    clearInterval(etaInterval);
    etaInterval = null;
  }
}

// ---------------------------------------------------------------------------
// Simple HTML escape
// ---------------------------------------------------------------------------

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Autocomplete search
// ---------------------------------------------------------------------------

searchInput.addEventListener("input", () => {
  const query = searchInput.value.trim();
  resultsList.innerHTML = "";
  selectedSymbol = "";

  if (timeout) {
    clearTimeout(timeout);
  }

  if (!query) return;

  timeout = setTimeout(async () => {
    try {
      const apiUrl = await getConfiguredApiUrl();
      const res = await fetch(`${apiUrl}/search?q=${encodeURIComponent(query)}`);
      const data = await res.json();

      resultsList.innerHTML = "";

      for (let i = 0; i < data.length; i++) {
        const stock = data[i];
        const li = document.createElement("li");
        li.textContent = `${stock.name} (${stock.symbol})`;

        li.onclick = () => {
          searchInput.value = stock.symbol;
          selectedSymbol = stock.symbol;
          resultsList.innerHTML = "";
        };

        resultsList.appendChild(li);
      }
    } catch (err) {
      console.error("Search failed", err);
    }
  }, 300);
});

// --- Hide dropdown when clicking outside ---
document.addEventListener("click", (event) => {
  if (!searchContainer.contains(event.target)) {
    resultsList.innerHTML = "";
  }
});

// ---------------------------------------------------------------------------
// Fetch prediction (with ETA pre-check)
// ---------------------------------------------------------------------------

predictBtn.addEventListener("click", async () => {
  const symbol = selectedSymbol || searchInput.value.trim().toUpperCase();
  if (!symbol) return;

  // Pre-check: does a model already exist?
  const availableStocks = await fetchAvailableStocks();
  const hasModel = availableStocks.includes(symbol);

  if (hasModel) {
    setBtnLoading(true, "Predicting...");
    resultsDiv.innerHTML = `
      <p style="color:#2563eb;">Predicting <strong>${symbol}</strong>...</p>
      <p style="color:#9ca3af;font-size:12px;" id="eta-display">ETA: calculating...</p>
    `;
    startEtaTimer(5);
  } else {
    setBtnLoading(true, "Training...");
    resultsDiv.innerHTML = `
      <p style="color:#d97706;">Training model for <strong>${symbol}</strong>...</p>
      <p style="color:#9ca3af;font-size:12px;">No existing model found. Training from scratch.</p>
      <p style="color:#9ca3af;font-size:12px;" id="eta-display">ETA: calculating...</p>
    `;
    startEtaTimer(60);
  }

  try {
    const bgResponse = await new Promise((resolve, reject) => {
      chrome.runtime.sendMessage({ type: "runPrediction", symbol, hasModel }, (response) => {
        const runtimeError = chrome.runtime.lastError;
        if (runtimeError) {
          reject(new Error(runtimeError.message));
          return;
        }
        resolve(response);
      });
    });

    if (!bgResponse || !bgResponse.ok || !bgResponse.data) {
      throw new Error(bgResponse?.error || "Prediction failed");
    }

    setBtnLoading(false);
    const symbolCurrency = await getSymbolCurrency(symbol);
    resultsDiv.innerHTML = renderPredictionHtml(bgResponse.data, symbolCurrency);

    chrome.storage.local.get({ stockUsage: {} }, (res) => {
      const usage = res.stockUsage;
      usage[symbol] = { lastUsed: Date.now() };
      chrome.storage.local.set({ stockUsage: usage }, renderSavedStocks);
    });

  } catch (err) {
    setBtnLoading(false);
    resultsDiv.innerHTML = `<p style="color:#dc2626;">Error: ${escapeHtml(err.message)}</p>`;
  }
});

// --- Listen for state changes from background ---
chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName !== "local" || !changes.analysisState) return;
  restoreAnalysisState();
});

// --- Render saved stocks (top 5 by last used) ---
async function renderSavedStocks() {
  let stockUsage = {};

  await new Promise(resolve => {
    chrome.storage.local.get({ stockUsage: {} }, (res) => {
      stockUsage = res.stockUsage;
      resolve();
    });
  });

  let allStocks = [];
  try {
    const apiUrl = await getConfiguredApiUrl();
    const res = await fetch(`${apiUrl}/available_stocks`);
    const data = await res.json();
    allStocks = data.stocks || [];
  } catch (err) {
    console.error("Failed to fetch available stocks", err);
  }

  const savedStocks = [];
  for (let i = 0; i < allStocks.length; i++) {
    const sym = allStocks[i];
    const lastUsed = stockUsage[sym]?.lastUsed || 0;
    savedStocks.push({ symbol: sym, lastUsed: lastUsed });
  }

  savedStocks.sort((a, b) => b.lastUsed - a.lastUsed);

  savedDiv.innerHTML = "";
  for (let i = 0; i < Math.min(5, savedStocks.length); i++) {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.textContent = savedStocks[i].symbol;
    chip.onclick = () => {
      searchInput.value = savedStocks[i].symbol;
      selectedSymbol = savedStocks[i].symbol;
    };
    savedDiv.appendChild(chip);
  }
}

// ---------------------------------------------------------------------------
// Server toggle
// ---------------------------------------------------------------------------

const serverSwitch = document.getElementById("serverSwitch");
const toggleLabel = document.getElementById("toggleLabel");

function updateToggleUI(url) {
  const isLocal = url === CONFIG.LOCAL_URL;
  serverSwitch.checked = !isLocal; // right (checked) = deployed, left (unchecked) = local
  toggleLabel.textContent = isLocal ? "Local" : "Render";
  toggleLabel.style.color = isLocal ? "#16a34a" : "#2563eb";
}

serverSwitch.addEventListener("change", async () => {
  const newUrl = serverSwitch.checked ? CONFIG.DEPLOYED_URL : CONFIG.LOCAL_URL;
  toggleLabel.textContent = "Connecting...";
  toggleLabel.style.color = "#6b7280";

  try {
    await setActiveServer(newUrl);
    updateToggleUI(newUrl);
    // Refresh saved stocks list for the new server
    renderSavedStocks();
  } catch (err) {
    console.error("Failed to switch server", err);
    toggleLabel.textContent = "Error";
    toggleLabel.style.color = "#dc2626";
  }
});

async function initServerToggle() {
  const url = await getConfiguredApiUrl();
  updateToggleUI(url);
}

// --- Init ---
renderSavedStocks();
restoreAnalysisState();
initServerToggle();
