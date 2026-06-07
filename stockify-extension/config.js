// Configuration file for Stockify Extension
// Supports both local development server and deployed Render server

const CONFIG = {
  LOCAL_URL: "http://127.0.0.1:8000",
  DEPLOYED_URL: "https://stockify-server-90a0.onrender.com",
  API_URL: null // Will be resolved dynamically
};

// Cache the resolved server URL
let _resolvedApiUrl = null;
let _resolving = false;
let _resolvers = [];

/**
 * Check if a server is reachable by hitting a lightweight endpoint.
 */
function _probeServer(url) {
  return new Promise((resolve) => {
    const timeout = new AbortController();
    const timer = setTimeout(() => {
      timeout.abort();
      resolve(false);
    }, 3000);

    fetch(`${url}/available_stocks`, { method: "GET", signal: timeout.signal })
      .then((res) => {
        clearTimeout(timer);
        resolve(res.ok);
      })
      .catch(() => {
        clearTimeout(timer);
        resolve(false);
      });
  });
}

/**
 * Detect which server is available (local first, then deployed).
 * Result is cached in chrome.storage.local for future use.
 */
async function _detectServer() {
  // Check cache first
  const cached = await new Promise((resolve) => {
    if (chrome?.storage?.local) {
      chrome.storage.local.get({ serverUrl: null }, (res) => resolve(res.serverUrl));
    } else {
      resolve(null);
    }
  });

  if (cached) return cached;

  // Try local server first
  const localUp = await _probeServer(CONFIG.LOCAL_URL);
  if (localUp) return CONFIG.LOCAL_URL;

  // Fall back to deployed server
  const deployedUp = await _probeServer(CONFIG.DEPLOYED_URL);
  if (deployedUp) return CONFIG.DEPLOYED_URL;

  // Default to deployed if neither responds (e.g. both temporarily down)
  return CONFIG.DEPLOYED_URL;
}

/**
 * Set and persist the active server URL.
 */
async function setActiveServer(url) {
  _resolvedApiUrl = url;
  CONFIG.API_URL = url;
  if (chrome?.storage?.local) {
    chrome.storage.local.set({ serverUrl: url });
  }
}

/**
 * Returns the active API URL. On first call, detects the server
 * (or loads from cache) and resolves it.
 */
async function getConfiguredApiUrl() {
  if (_resolvedApiUrl) return _resolvedApiUrl;

  if (_resolving) {
    return new Promise((resolve) => _resolvers.push(resolve));
  }

  _resolving = true;
  try {
    const url = await _detectServer();
    _resolvedApiUrl = url;
    CONFIG.API_URL = url;

    // Persist for next time
    if (chrome?.storage?.local) {
      chrome.storage.local.set({ serverUrl: url });
    }

    // Resolve any waiting callers
    _resolvers.forEach((r) => r(url));
    _resolvers = [];

    return url;
  } finally {
    _resolving = false;
  }
}

// Kick off detection so CONFIG.API_URL is ready by the time it's needed
getConfiguredApiUrl();