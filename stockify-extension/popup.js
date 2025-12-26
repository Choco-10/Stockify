const API_URL = "http://127.0.0.1:8000";


const searchInput = document.getElementById("stock-search");
const resultsList = document.getElementById("search-results");
const predictBtn = document.getElementById("predictBtn");
const resultsDiv = document.getElementById("results");
const savedDiv = document.getElementById("savedStocks");

const searchContainer = document.querySelector(".search-container");

let selectedSymbol = "";
let timeout = null;

// --- Autocomplete search ---
searchInput.addEventListener("input", () => {
  const query = searchInput.value.trim();
  resultsList.innerHTML = "";
  selectedSymbol = "";

  if (timeout) {
    clearTimeout(timeout);
  }

  if (!query) {
    return;
  }

  timeout = setTimeout(async () => {
    try {
      const res = await fetch(`${API_URL}/search?q=${encodeURIComponent(query)}`);
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

// --- Utility to map currency code to symbol ---
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
    // add more if needed
  };
  return map[currency] || currency;
}

// --- Fetch prediction ---
predictBtn.addEventListener("click", async () => {
  const symbol = selectedSymbol || searchInput.value.trim().toUpperCase();
  if (!symbol) return;

  resultsDiv.innerHTML = "Loading...";

  try {
    // 1️⃣ Fetch prediction
    const res = await fetch(`${API_URL}/predict?symbol=${symbol}`);
    if (!res.ok) throw new Error("Prediction failed");
    const data = await res.json();

    // 2️⃣ Fetch updated master stocks to get correct currency
    const stockRes = await fetch(`${API_URL}/search?q=${encodeURIComponent(symbol)}`);
    const stockList = await stockRes.json();
    const stockData = stockList.find(s => s.symbol.toUpperCase() === symbol.toUpperCase());
    const symbolCurrency = getCurrencySymbol(stockData?.currency || "USD");

    // 3️⃣ Display prediction with correct currency symbol
    resultsDiv.innerHTML = `
      <p>Symbol: ${data.symbol}</p>
      <p>Current Price: ${symbolCurrency}${data.current_price.toFixed(2)}</p>
      <p>Next Day Prediction: ${symbolCurrency}${data.next_day_prediction.toFixed(2)}</p>
    `;

    // 4️⃣ Update stock usage timestamp
    chrome.storage.local.get({ stockUsage: {} }, (res) => {
      const usage = res.stockUsage;
      usage[symbol] = { lastUsed: Date.now() };
      chrome.storage.local.set({ stockUsage: usage }, renderSavedStocks);
    });

  } catch (err) {
    resultsDiv.innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
  }
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
    const res = await fetch(`${API_URL}/available_stocks`);
    const data = await res.json();
    allStocks = data.stocks;
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

// --- Init ---
renderSavedStocks();
