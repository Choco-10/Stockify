# Stockify

**AI-powered stock price prediction** — Stockify uses LSTM neural networks to forecast next-day stock price movements. It predicts whether a stock will go **UP**, **DOWN**, or stay **FLAT**, along with the expected return percentage and confidence score. The project includes a FastAPI backend for model training and inference, and a Chrome browser extension for a seamless user experience.

<a href="https://www.freepik.com/icon/bar-chart_1271708#fromView=search&page=1&position=20&uuid=1182db69-f906-403f-a79e-16cf3e29f41c">Icon by srip</a>

---

## Features

- **LSTM Ensemble Predictions** — Trains 3 LSTM models with varying lookback windows (30, 60, and 90 days) and combines their outputs using confidence-weighted voting for more robust predictions.
- **On-the-Fly Training** — If no trained model exists for a requested stock symbol, the system automatically downloads historical data from Yahoo Finance and trains a model from scratch (~1 minute).
- **Stock Search with Fuzzy Matching** — Search for stocks by company name or ticker symbol. Uses `rapidfuzz` for prefix and fuzzy matching, returning the top 5 closest results.
- **Chrome Extension UI** — A Manifest V3 Chrome extension with a popup interface to search stocks, trigger predictions, and view results directly in your browser.

---

## How It Works

1. **Data Collection** — Historical daily stock data (Open, High, Low, Close, Volume) is fetched from Yahoo Finance via the `yfinance` library.
2. **Feature Engineering** — Raw OHLCV data is combined with technical indicators to create a rich feature set for the model.
3. **Sequence Generation** — Data is converted into fixed-length sequences (30, 60, or 90 days) using a sliding window approach.
4. **LSTM Training** — A multi-layer LSTM network is trained on the sequences to predict the next day's price direction.
5. **Ensemble Aggregation** — Predictions from all 3 sequence lengths are combined using confidence-weighted voting.
6. **Output** — The system returns the predicted direction (UP/DOWN/FLAT), expected return percentage, predicted closing price, and model confidence.

---

## Getting Started

### Prerequisites

- **Python 3.8+**
- **Google Chrome** (for the browser extension)

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Choco-10/Stockify.git
   cd Stockify
   ```

2. **Install Python dependencies**
   ```bash
   cd server
   pip install -r requirements.txt
   ```

3. **Start the FastAPI server**
   ```bash
   uvicorn app:app --reload
   ```
   The server will be available at `http://127.0.0.1:8000`.

4. **Verify the server is running**
   Open `http://127.0.0.1:8000/docs` in your browser to see the interactive API documentation (Swagger UI).

### Chrome Extension Setup

1. Open Google Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in the top-right corner)
3. Click **Load unpacked** and select the `stockify-extension/` folder
4. The Stockify icon will appear in your browser toolbar
5. Click the icon, search for a stock, and hit **Predict**

> **Note:** By default the extension connects to the local server (`http://127.0.0.1:8000`). To use the production server, edit `stockify-extension/config.js` and uncomment the Render URL.

---

## Disclaimer

Stockify is built for educational and research purposes only. It should **not** be used as financial advice. Stock markets are inherently unpredictable, and past performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.