import json
import os
import shutil
import traceback
from time import sleep

from train import train_new_stock
from config import DATA_DIR, MODELS_DIR


def rm_glob(dirpath, patterns):
    if not os.path.exists(dirpath):
        return
    for name in os.listdir(dirpath):
        path = os.path.join(dirpath, name)
        for p in patterns:
            if name.endswith(p):
                try:
                    os.remove(path)
                except Exception:
                    try:
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                    except Exception:
                        pass


def main():
    stocks_file = os.path.join(os.path.dirname(__file__), "stocks", "master_stocks.json")
    if not os.path.exists(stocks_file):
        raise RuntimeError(f"Missing stock list: {stocks_file}")

    with open(stocks_file, "r", encoding="utf-8") as f:
        stocks = json.load(f)

    symbols = [s["symbol"].upper() for s in stocks]

    # Clean data and models
    print("Cleaning data directory and model artifacts...")
    rm_glob(DATA_DIR, [".csv"])
    rm_glob(MODELS_DIR, [".onnx", "_scaler.json", ".json"])

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    results = []
    for sym in symbols:
        print(f"Processing {sym}...")
        try:
            # train_new_stock will fetch data and export ONNX/scaler
            train_new_stock(sym)
            results.append({"symbol": sym, "status": "ok"})
        except Exception as e:
            traceback.print_exc()
            results.append({"symbol": sym, "status": "error", "error": str(e)})
        # brief pause to avoid hammering APIs
        sleep(1.0)

    print("Rebuild complete. Summary:")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
