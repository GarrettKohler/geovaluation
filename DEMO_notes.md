  ### The notebook trains two HistGradientBoosting models (sklearn's histogram-based gradient boosting, similar to XGBoost/LightGBM) on the same 52 features and  same 26,099 active sites — but they answer fundamentally different questions.                                                                              
                                                                                                                                                             
  ---                                                                                                                                                        
  Model 1: Regression (HistGradientBoostingRegressor)                                                                                                      

  Question it answers: "How much monthly revenue will this site generate?"

  - Target: avg_monthly_revenue — a continuous dollar value
  - Loss function: squared_error (standard MSE)
  - Output during inference: A dollar amount per site
    - e.g., $143.50, $612.00, $1,247.83
    - The full revenue curve — predicts low, medium, AND high earners

  Test performance: R² = 0.45, RMSE = $212, MAE = $134

  ---
  Model 2: Classification (HistGradientBoostingClassifier)

  Question it answers: "Is this site a top performer (p90+)?"

  - Target: Binary label — 1 if avg_monthly_revenue >= $613 (the 90th percentile), 0 otherwise
  - Loss function: log_loss (binary cross-entropy)
  - Output during inference: Two things:
    - predict() → a binary label: 1 (top performer) or 0 (not)
    - predict_proba() → a probability score between 0.0 and 1.0
        - e.g., 0.87 = 87% likelihood of being a top performer
      - This is what the "lookalike" model in the main app uses for site ranking

  Test performance: AUC = 0.87, F1 = 0.47, Precision = 0.68, Recall = 0.36

  ---
  Side-by-Side Inference Example

  ┌────────┬───────────────────┬───────────────────────────────────────┐
  │  Site  │ Regression Output │         Classification Output         │
  ├────────┼───────────────────┼───────────────────────────────────────┤
  │ Site A │ $742/mo           │ 0.91 probability (predicted: Top)     │
  ├────────┼───────────────────┼───────────────────────────────────────┤
  │ Site B │ $180/mo           │ 0.03 probability (predicted: Not Top) │
  ├────────┼───────────────────┼───────────────────────────────────────┤
  │ Site C │ $595/mo           │ 0.44 probability (predicted: Not Top) │
  └────────┴───────────────────┴───────────────────────────────────────┘

  Site C is the interesting case — it's close to the $613 threshold. Regression says "almost there" ($595), but the classifier says "probably not" (44%). The
   notebook's agreement analysis (cell 10) shows 97.6% agreement between the two models, with the 2.4% disagreement concentrated around this threshold
  boundary.

  ★ Insight ─────────────────────────────────────
  1. Regression learns the entire revenue distribution — it cares about the difference between a $50 site and a $150 site just as much as between $500 and
  $600. This makes it useful for revenue forecasting and budget planning, but it struggles with the long tail (high-revenue outliers pull the loss up, hence
  the $212 RMSE).
  2. Classification only learns the decision boundary at p90 — it ignores distinctions below the threshold entirely. A $50 site and a $500 site are both just
   "0". This makes it better for the action question ("should we invest in this site?") but throws away information about how much revenue to expect.
  3. The probability output from classification is the more useful inference artifact in practice — rather than the hard 0/1 label, the predict_proba() score
   lets you rank all sites by "top-performer-ness" and pick your own threshold. That's exactly how the main app's lookalike model works with configurable
  percentile bounds.
  ─────────────────────────────────────────────────
