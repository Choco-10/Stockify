# Model Efficiency Testing

This folder helps you answer two questions:

1. Is the model predicting price reasonably well?
2. Is the model fast enough for practical use?

The test script measures both accuracy and speed in one run.

## What this test does in simple words

Think of this as a quiz for the model:

1. Pick the most recent days from history (for example, last 120 days).
2. For each day in that period, give the model only the past 60 days.
3. Ask it to predict the next close price.
4. Compare prediction with the real close price.
5. Record how long each prediction took.

This is called walk-forward testing. It is better than random splitting for stock data because time order matters.

## Why this method is used

- Stock data is sequential, so future values must never leak into past input.
- The method matches real usage: at prediction time, you only know the past.
- It produces metrics you can track over time after retraining.

## Inputs required

The script expects these files:

- Model file: models/SYMBOL.onnx
- Scaler metadata: models/SYMBOL_scaler.json
- Historical prices: data/SYMBOL.csv

If any of these are missing for a symbol, that symbol is marked as skipped in the report.

## Metrics explained

### Accuracy metrics

- MAE
Average absolute error in price units.
Example: MAE = 6 means predictions are off by about 6 currency units on average.

- RMSE
Similar to MAE, but penalizes large mistakes more strongly.
If RMSE is much higher than MAE, the model makes occasional big misses.

- MAPE percent
Average percentage error.
Easy to compare across stocks with different price ranges.

- sMAPE percent
Balanced percentage error that is more stable than MAPE in some cases.

- directional_accuracy_percent
How often the model got direction right (up or down versus previous day).
50 percent is around random direction guessing for balanced movement.

### Speed metrics

- avg_latency_ms
Average time for one prediction.

- p95_latency_ms
95 percent of predictions are faster than this number.
Useful for worst-case behavior.

- throughput_preds_per_sec
Approximate predictions per second.

## How to run

Run from project root:

```powershell
python testing/evaluate_efficiency.py --test-size 120
```

Run selected symbols only:

```powershell
python testing/evaluate_efficiency.py --symbols AAPL MSFT NVDA --test-size 180
```

Optional arguments:

- --test-size
Number of latest days used for evaluation.

- --warmup
Initial predictions excluded from latency stats to reduce startup noise.

## Output files

Each run does two outputs:

1. Prints JSON in terminal.
2. Saves JSON report in testing/reports with timestamped filename.

Example output file:

- testing/reports/efficiency_report_YYYYMMDD_HHMMSS.json

## How to read results quickly

Use this quick checklist:

1. MAPE percent lower is better.
2. directional_accuracy_percent above 50 percent is usually a good sign.
3. Compare MAE against typical daily movement of that stock.
4. Ensure latency numbers are acceptable for your API needs.

## Expected metric ranges (practical targets)

For daily next-close stock prediction, these ranges are realistic in many real-world setups:

- MAPE percent
Poor: above 5.0
Acceptable: 3.0 to 5.0
Good: 1.5 to 3.0
Very strong: below 1.5

- directional_accuracy_percent
Poor: below 50
Acceptable: 50 to 54
Good: 55 to 59
Very strong: 60+

- RMSE vs MAE
Healthy sign: RMSE is not much larger than MAE.
Warning sign: RMSE much larger than MAE means occasional large misses.

- avg_latency_ms (for API usage)
Excellent: below 5 ms
Good: 5 to 20 ms
Acceptable: 20 to 50 ms
Slow: above 50 ms

Important:
- Stocks are noisy and non-stationary. Even good models can degrade over time.
- Direction accuracy above 60 percent is difficult and often not stable across long periods.
- A model can have low MAPE but still poor direction accuracy, and vice versa.

## How to judge your current result

Example from your latest run:

- MAPE percent around 2.33 means price error is in the good range.
- directional_accuracy_percent at 45 means direction is currently poor.
- avg_latency_ms around 2.76 means speed is excellent.

So right now: price level prediction is decent, speed is strong, but direction prediction needs improvement.

## Recommended baseline comparisons

To know if the model is truly useful, compare with simple baselines:

- Naive baseline: tomorrow equals today.
- Moving average baseline: use mean of recent closes.

If your model does not beat these on MAPE and direction, retraining or feature improvements are needed.

## Common issues

- Module import error when running script:
Run from project root so project imports resolve correctly.

- Symbol skipped in report:
Check that ONNX, scaler JSON, and CSV all exist for that symbol.

- Very slow first few predictions:
Normal due to runtime warmup; use warmup setting for fair latency reporting.
