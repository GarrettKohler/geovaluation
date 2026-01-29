# Conformal Prediction & Counterfactual Explanations: Implementation Strategy

## Executive Summary

This document outlines a strategy to extend the existing site scoring system with:
1. **Conformal Prediction (MAPIE)** - Uncertainty quantification with statistical guarantees
2. **Probability Calibration** - Ensures predicted probabilities match observed frequencies
3. **Counterfactual Explanations (DiCE)** - Actionable "what would need to change" insights
4. **Fleet-wide Intervention Clustering** - Strategic upgrade path identification

---

## Current Architecture vs. Proposed Extensions

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           CURRENT ARCHITECTURE                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │   Raw Data   │───▶│  Data Loader    │───▶│ SiteScoringModel │───▶│  Predictions  │  │
│  │  (Parquet)   │    │  (Processor)    │    │  (PyTorch MLP)   │    │   (Scores)    │  │
│  └──────────────┘    └─────────────────┘    └──────────────────┘    └───────────────┘  │
│                                                      │                                  │
│                                                      ▼                                  │
│                                             ┌──────────────────┐                        │
│                                             │  SHAP Analysis   │                        │
│                                             │ (Feature Import) │                        │
│                                             └──────────────────┘                        │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           PROPOSED EXTENSIONS                                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │   Raw Data   │───▶│  Data Loader    │───▶│ SiteScoringModel │───▶│  Predictions  │  │
│  │  (Parquet)   │    │  (Processor)    │    │  (PyTorch MLP)   │    │   (Scores)    │  │
│  └──────────────┘    └─────────────────┘    └──────────────────┘    └───────┬───────┘  │
│                              │                       │                      │           │
│                              │                       │                      ▼           │
│                              │                       │             ┌─────────────────┐  │
│                              ▼                       ▼             │  ★ CALIBRATOR   │  │
│                      ┌──────────────┐       ┌──────────────┐      │ (Isotonic Reg)  │  │
│                      │ Calibration  │       │ SHAP Analysis│      └────────┬────────┘  │
│                      │    Split     │       │              │               │           │
│                      │   (15-20%)   │       └──────────────┘               ▼           │
│                      └──────┬───────┘                              ┌─────────────────┐  │
│                             │                                      │  Calibrated     │  │
│                             ▼                                      │  Probabilities  │  │
│                     ┌──────────────────┐                           └────────┬────────┘  │
│                     │  ★ MAPIE        │                                     │           │
│                     │  (Conformal     │◀────────────────────────────────────┘           │
│                     │   Prediction)   │                                                 │
│                     └────────┬─────────┘                                                │
│                              │                                                          │
│                              ▼                                                          │
│         ┌────────────────────┴────────────────────┐                                     │
│         │                                         │                                     │
│         ▼                                         ▼                                     │
│ ┌──────────────────┐                     ┌──────────────────┐                          │
│ │ Prediction Sets  │                     │  ★ DiCE          │                          │
│ │ with Confidence  │                     │  (Counterfactual │                          │
│ │ Intervals        │                     │   Generation)    │                          │
│ └────────┬─────────┘                     └────────┬─────────┘                          │
│          │                                        │                                     │
│          │                                        ▼                                     │
│          │                               ┌──────────────────┐                          │
│          │                               │ ★ CLUSTERING     │                          │
│          │                               │ (Fleet Upgrade   │                          │
│          │                               │  Path Analysis)  │                          │
│          │                               └────────┬─────────┘                          │
│          │                                        │                                     │
│          ▼                                        ▼                                     │
│ ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│ │                         EXECUTIVE DASHBOARD                                        │ │
│ │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐  │ │
│ │  │ Tier Confidence │  │ Historical      │  │ Upgrade Paths   │  │ Fleet-wide    │  │ │
│ │  │ (8/10 succeed)  │  │ Accuracy Bands  │  │ (Actionable)    │  │ Interventions │  │ │
│ │  └─────────────────┘  └─────────────────┘  └─────────────────┘  └───────────────┘  │ │
│ └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
│  ★ = NEW COMPONENT                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Complete Pipeline

```
                                    DATA FLOW ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════════════════

PHASE 1: DATA PREPARATION & MODEL TRAINING
──────────────────────────────────────────────────────────────────────────────────────────

    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                        site_training_data.parquet                                │
    │                           (~26K active sites)                                    │
    └───────────────────────────────────────┬─────────────────────────────────────────┘
                                            │
                            ┌───────────────┼───────────────┐
                            ▼               ▼               ▼
                    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
                    │   TRAIN     │ │    VAL      │ │    TEST     │
                    │    70%      │ │    15%      │ │    15%      │
                    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
                           │               │               │
                           │        ┌──────┴──────┐        │
                           │        │ Split for   │        │
                           │        │ CALIBRATION │        │
                           │        │  (50% val)  │        │
                           │        └──────┬──────┘        │
                           │               │               │
                           ▼               ▼               ▼
                    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
                    │ TRAIN       │ │ CALIBRATION │ │ HOLDOUT     │
                    │ (17,500)    │ │  (~1,950)   │ │  (~3,900)   │
                    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
                           │               │               │
                           ▼               │               │
    ┌─────────────────────────────────┐    │               │
    │     SiteScoringModel            │    │               │
    │   ───────────────────────       │    │               │
    │   • Embedding + ResidualMLP     │    │               │
    │   • Feature Selection (STG)     │    │               │
    │   • Early Stopping              │    │               │
    └─────────────────┬───────────────┘    │               │
                      │                    │               │
                      ▼                    ▼               │
            ┌──────────────────────────────────────────┐   │
            │         CALIBRATION PHASE                │   │
            │  ┌────────────────────────────────────┐  │   │
            │  │  1. Get model predictions on CAL   │  │   │
            │  │  2. Fit IsotonicRegression         │  │   │
            │  │  3. Fit MAPIE (conformal)          │  │   │
            │  └────────────────────────────────────┘  │   │
            └─────────────────┬────────────────────────┘   │
                              │                            │
                              ▼                            ▼
                    ┌───────────────────────────────────────────┐
                    │          EVALUATION ON HOLDOUT            │
                    │  • Calibrated probabilities               │
                    │  • Prediction sets with confidence        │
                    │  • Coverage verification                  │
                    └───────────────────────────────────────────┘


PHASE 2: INFERENCE & EXPLANATION
──────────────────────────────────────────────────────────────────────────────────────────

                        ┌─────────────────────────────┐
                        │     NEW SITE / QUERY        │
                        │    (single or batch)        │
                        └─────────────┬───────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                 │
    │                        EXPLAINABILITY PIPELINE                                  │
    │                                                                                 │
    │  ┌─────────────────────────────────────────────────────────────────────────┐   │
    │  │  STEP 1: Base Prediction                                                │   │
    │  │  ──────────────────────                                                 │   │
    │  │  raw_score = model.predict(site_features)                               │   │
    │  └────────────────────────────────────┬────────────────────────────────────┘   │
    │                                       │                                        │
    │                                       ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────────────┐   │
    │  │  STEP 2: Probability Calibration                                        │   │
    │  │  ───────────────────────────────                                        │   │
    │  │  calibrated_prob = calibrator.predict(raw_score)                        │   │
    │  │                                                                         │   │
    │  │  Output: "75% probability" means ~75% of similar sites succeeded        │   │
    │  └────────────────────────────────────┬────────────────────────────────────┘   │
    │                                       │                                        │
    │                                       ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────────────┐   │
    │  │  STEP 3: Conformal Prediction (MAPIE)                                   │   │
    │  │  ────────────────────────────────────                                   │   │
    │  │  y_pred, y_sets = mapie.predict(site_features, alpha=[0.10])            │   │
    │  │                                                                         │   │
    │  │  Output: Prediction set at 90% confidence                               │   │
    │  │  - {0} = confident LOW value                                            │   │
    │  │  - {1} = confident HIGH value                                           │   │
    │  │  - {0,1} = uncertain (needs more info)                                  │   │
    │  └────────────────────────────────────┬────────────────────────────────────┘   │
    │                                       │                                        │
    │                                       ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────────────┐   │
    │  │  STEP 4: SHAP Feature Importance                                        │   │
    │  │  ───────────────────────────────                                        │   │
    │  │  shap_values = explainer(site_features)                                 │   │
    │  │                                                                         │   │
    │  │  Output: "This site scored low because:                                 │   │
    │  │           - Low active_months (-0.15)                                   │   │
    │  │           - Far from interstate (-0.08)                                 │   │
    │  │           + High household income (+0.12)"                              │   │
    │  └────────────────────────────────────┬────────────────────────────────────┘   │
    │                                       │                                        │
    │                           ┌───────────┴───────────┐                           │
    │                           │ Is prediction LOW?    │                           │
    │                           └───────────┬───────────┘                           │
    │                                       │                                        │
    │                         ┌─────────────┴─────────────┐                         │
    │                         │ YES                   NO  │                         │
    │                         ▼                       ▼   │                         │
    │  ┌──────────────────────────────────┐    ┌─────────────────────────────────┐  │
    │  │  STEP 5: Counterfactual (DiCE)   │    │  No counterfactuals needed     │  │
    │  │  ─────────────────────────────   │    │  (already high-value)          │  │
    │  │                                  │    └─────────────────────────────────┘  │
    │  │  counterfactuals = dice.generate(│                                        │
    │  │    site,                         │                                        │
    │  │    desired_class=1,              │                                        │
    │  │    features_to_vary=ACTIONABLE,  │                                        │
    │  │    permitted_range={...}         │                                        │
    │  │  )                               │                                        │
    │  │                                  │                                        │
    │  │  Output: "To upgrade this site:  │                                        │
    │  │   Option 1: Extend hours to 18h  │                                        │
    │  │   Option 2: Add 2 maintenance    │                                        │
    │  │             visits per month"    │                                        │
    │  └──────────────────────────────────┘                                        │
    │                                                                                 │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │         EXECUTIVE OUTPUT            │
                    │  ───────────────────────────        │
                    │  Tier: PROMISING                    │
                    │  Confidence: 8/10 similar succeed   │
                    │  Key Drivers: [SHAP waterfall]      │
                    │  Upgrade Path: Extend to 18h ops    │
                    └─────────────────────────────────────┘
```

---

## Module Structure

```
site_scoring/
├── model.py                    # Existing: SiteScoringModel
├── config.py                   # Existing: Config dataclass
├── data_loader.py              # Existing: Data processing
│
├── feature_selection/          # Existing: STG, LassoNet, etc.
│   ├── __init__.py
│   ├── stochastic_gates.py
│   ├── lassonet.py
│   └── ...
│
└── explainability/             # ★ NEW MODULE
    ├── __init__.py             # Exports: ExplainabilityPipeline, TierClassifier
    │
    ├── conformal.py            # ★ MAPIE wrapper for conformal prediction
    │   │
    │   │  Classes:
    │   │  ─────────
    │   │  • ConformalClassifier
    │   │      - __init__(model, calibration_data, alpha=0.10)
    │   │      - fit(X_cal, y_cal)
    │   │      - predict_sets(X)  → (predictions, confidence_sets)
    │   │      - get_coverage_stats()  → coverage verification
    │   │
    │   │  • ConformalRegressor (for revenue prediction)
    │   │      - predict_intervals(X, alpha)  → (y_pred, y_lower, y_upper)
    │   │
    │   └───────────────────────────────────────────────────────────────
    │
    ├── calibration.py          # ★ Probability calibration
    │   │
    │   │  Classes:
    │   │  ─────────
    │   │  • ProbabilityCalibrator
    │   │      - __init__(method='isotonic')  # or 'platt'
    │   │      - fit(y_proba_uncalibrated, y_true)
    │   │      - calibrate(y_proba)  → calibrated_proba
    │   │      - plot_reliability_diagram()
    │   │      - get_brier_score()
    │   │
    │   └───────────────────────────────────────────────────────────────
    │
    ├── counterfactuals.py      # ★ DiCE integration
    │   │
    │   │  Classes:
    │   │  ─────────
    │   │  • CounterfactualGenerator
    │   │      - __init__(model, train_data, actionable_features, immutable_features)
    │   │      - generate(site, n_counterfactuals=5, desired_class=1)
    │   │      - generate_batch(sites)  → List[Counterfactual]
    │   │      - get_upgrade_paths()  → summarized changes
    │   │
    │   │  • UpgradePathClusterer
    │   │      - __init__(n_clusters=5)
    │   │      - fit(counterfactual_changes)
    │   │      - get_fleet_recommendations()  → strategic interventions
    │   │
    │   └───────────────────────────────────────────────────────────────
    │
    ├── tiers.py                # ★ Executive-friendly tier classification
    │   │
    │   │  Classes:
    │   │  ─────────
    │   │  • TierClassifier
    │   │      - __init__(thresholds=[0.85, 0.65, 0.50])
    │   │      - classify(calibrated_prob)  → Tier
    │   │      - get_tier_accuracy()  → historical accuracy by tier
    │   │      - get_business_label(tier)  → "Recommended", "Promising", etc.
    │   │
    │   │  Constants:
    │   │  ──────────
    │   │  • TIER_LABELS = {1: "Recommended", 2: "Promising", 3: "Review", 4: "Not Recommended"}
    │   │  • TIER_ACTIONS = {1: "Proceed to contract", 2: "Site visit", ...}
    │   │
    │   └───────────────────────────────────────────────────────────────
    │
    └── pipeline.py             # ★ Unified explainability interface
        │
        │  Classes:
        │  ─────────
        │  • ExplainabilityPipeline
        │      - __init__(model, train_data, actionable_features)
        │      - fit_calibration(X_cal, y_cal)
        │      - explain_site(site_features)  → ExplanationResult
        │      - explain_batch(sites)  → List[ExplanationResult]
        │      - get_fleet_interventions(low_value_sites)  → UpgradePaths
        │      - save(path) / load(path)
        │
        │  Dataclasses:
        │  ─────────────
        │  • ExplanationResult
        │      - prediction: float
        │      - calibrated_prob: float
        │      - confidence_set: Set[int]
        │      - tier: int
        │      - tier_label: str
        │      - shap_values: np.ndarray
        │      - top_drivers: List[Tuple[str, float]]
        │      - counterfactuals: Optional[List[Counterfactual]]
        │
        └───────────────────────────────────────────────────────────────
```

---

## Feature Classification for Counterfactuals

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           FEATURE CLASSIFICATION                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ╔═══════════════════════════════════════════════════════════════════════════════════╗  │
│  ║  IMMUTABLE FEATURES (Cannot be changed - excluded from counterfactuals)           ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                   ║  │
│  ║  Location/Demographics:                                                           ║  │
│  ║  • dma_rank (market tier - inherent to location)                                  ║  │
│  ║  • avg_household_income, median_age (census-based)                                ║  │
│  ║  • pct_female, pct_male (demographics)                                            ║  │
│  ║  • log_nearest_site_distance_mi (location-fixed)                                  ║  │
│  ║  • log_min_distance_to_interstate_mi                                              ║  │
│  ║  • nearest_interstate                                                             ║  │
│  ║                                                                                   ║  │
│  ║  Site Identity:                                                                   ║  │
│  ║  • network, program (contractual)                                                 ║  │
│  ║  • retailer (partner identity)                                                    ║  │
│  ║  • brand_fuel, brand_restaurant, brand_c_store                                    ║  │
│  ║                                                                                   ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                          │
│  ╔═══════════════════════════════════════════════════════════════════════════════════╗  │
│  ║  ACTIONABLE FEATURES (Can be changed - DiCE will suggest modifications)          ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                   ║  │
│  ║  Operations (can increase only - monotonic):                                      ║  │
│  ║  • active_months → permitted_range: [current, current+12]                         ║  │
│  ║  • screen_count → permitted_range: [current, current+5]                           ║  │
│  ║                                                                                   ║  │
│  ║  Experience/Content (can change freely):                                          ║  │
│  ║  • experience_type (static, video, interactive)                                   ║  │
│  ║  • hardware_type                                                                  ║  │
│  ║                                                                                   ║  │
│  ║  Capabilities (boolean toggles - can enable):                                     ║  │
│  ║  • c_emv_enabled_encoded (enable EMV payments)                                    ║  │
│  ║  • c_nfc_enabled_encoded (enable NFC)                                             ║  │
│  ║  • c_open_24_hours_encoded (extend hours)                                         ║  │
│  ║  • c_vistar_programmatic_enabled_encoded                                          ║  │
│  ║  • c_walk_up_enabled_encoded                                                      ║  │
│  ║                                                                                   ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                          │
│  ╔═══════════════════════════════════════════════════════════════════════════════════╗  │
│  ║  RESTRICTION FLAGS (Business rules - typically immutable or policy-dependent)     ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                   ║  │
│  ║  • r_lottery_encoded, r_government_encoded, etc.                                  ║  │
│  ║  • These are often contractual/legal - treat as IMMUTABLE                         ║  │
│  ║  • Exception: Some restrictions might be negotiable → move to ACTIONABLE          ║  │
│  ║                                                                                   ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Executive Output: Tier System

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           EXECUTIVE TIER CLASSIFICATION                                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│    Score Range          Business Label          Historical           Recommended        │
│    ───────────          ──────────────          Accuracy             Action             │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                                 │   │
│  │   > 0.85  ─────────▶  ██████████████████████  RECOMMENDED      Proceed to      │   │
│  │                       █  "8-9 out of 10     █   (88%)          contract        │   │
│  │                       █   similar sites     █                                  │   │
│  │                       █   succeeded"        █                                  │   │
│  │                       ██████████████████████                                   │   │
│  │                                                                                 │   │
│  ├─────────────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                                 │   │
│  │  0.65-0.85 ────────▶  ████████████████████    PROMISING        Site visit      │   │
│  │                       █  "7-8 out of 10    █   (76%)          required        │   │
│  │                       █   similar sites    █                                  │   │
│  │                       █   succeeded"       █                                  │   │
│  │                       ████████████████████                                    │   │
│  │                                                                                 │   │
│  ├─────────────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                                 │   │
│  │  0.50-0.65 ────────▶  ██████████████████      REVIEW           Detailed       │   │
│  │                       █  "6 out of 10     █   REQUIRED         assessment     │   │
│  │                       █   similar sites   █   (62%)            needed         │   │
│  │                       █   succeeded"      █                                   │   │
│  │                       ██████████████████                                      │   │
│  │                                                                                 │   │
│  ├─────────────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                                 │   │
│  │   < 0.50  ─────────▶  ████████████████        NOT              Do not         │   │
│  │                       █  "Less than 5   █   RECOMMENDED       pursue          │   │
│  │                       █   out of 10     █   (N/A)                             │   │
│  │                       █   succeed"      █                                     │   │
│  │                       ████████████████                                        │   │
│  │                                                                                 │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│  Note: Historical accuracy percentages are computed on holdout data after deployment    │
│        and should be updated quarterly as more predictions are validated.               │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Fleet-Wide Intervention Analysis

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    FLEET-WIDE UPGRADE PATH IDENTIFICATION                                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  STEP 1: Generate counterfactuals for all low-value sites                               │
│  ────────────────────────────────────────────────────────                               │
│                                                                                          │
│       Low-Value Sites              Counterfactuals                                      │
│       ────────────────             ────────────────                                     │
│            Site A  ─────────────▶  [CF1, CF2, CF3]                                      │
│            Site B  ─────────────▶  [CF1, CF2, CF3]                                      │
│            Site C  ─────────────▶  [CF1, CF2, CF3]                                      │
│              ...                        ...                                             │
│            Site N  ─────────────▶  [CF1, CF2, CF3]                                      │
│                                                                                          │
│  STEP 2: Extract change vectors (counterfactual - original)                             │
│  ──────────────────────────────────────────────────────────                             │
│                                                                                          │
│       change_vector = counterfactual_features - original_features                       │
│                                                                                          │
│       Example: Site A CF1                                                               │
│       ┌──────────────────────┬───────────┬──────────┬────────────┐                     │
│       │      Feature         │  Original │    CF    │   Change   │                     │
│       ├──────────────────────┼───────────┼──────────┼────────────┤                     │
│       │ active_months        │     6     │    12    │    +6      │                     │
│       │ screen_count         │     2     │     3    │    +1      │                     │
│       │ c_open_24_hours      │     0     │     1    │    +1      │                     │
│       │ experience_type      │  static   │  video   │   change   │                     │
│       └──────────────────────┴───────────┴──────────┴────────────┘                     │
│                                                                                          │
│  STEP 3: Cluster change vectors using K-Means                                           │
│  ─────────────────────────────────────────────                                          │
│                                                                                          │
│       ┌───────────────────────────────────────────────────────────────────────────┐    │
│       │                                                                           │    │
│       │     Cluster 1 (42 sites)          Cluster 2 (28 sites)                    │    │
│       │     ┌─────────────────┐          ┌─────────────────┐                      │    │
│       │     │  ● ●            │          │        ● ●      │                      │    │
│       │     │    ●  ●         │          │      ●   ●      │                      │    │
│       │     │  ●    ●  ●      │          │    ●  ●         │                      │    │
│       │     │     ●           │          │        ●        │                      │    │
│       │     └─────────────────┘          └─────────────────┘                      │    │
│       │     "Extend Hours"               "Add Screens"                            │    │
│       │                                                                           │    │
│       │     Cluster 3 (35 sites)          Cluster 4 (18 sites)                    │    │
│       │     ┌─────────────────┐          ┌─────────────────┐                      │    │
│       │     │   ●    ●        │          │  ●              │                      │    │
│       │     │      ●   ●      │          │    ●  ●         │                      │    │
│       │     │    ●      ●     │          │       ●         │                      │    │
│       │     │        ●        │          │                 │                      │    │
│       │     └─────────────────┘          └─────────────────┘                      │    │
│       │     "Enable Programmatic"        "Upgrade Hardware"                       │    │
│       │                                                                           │    │
│       └───────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  STEP 4: Interpret cluster centers as strategic interventions                           │
│  ─────────────────────────────────────────────────────────────                          │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                     STRATEGIC INTERVENTION SUMMARY                                 │ │
│  ├────────────────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                                    │ │
│  │  INTERVENTION A: "Extended Hours Initiative"                                       │ │
│  │  ─────────────────────────────────────────────                                     │ │
│  │  • Applies to: 42 sites (34% of low-value portfolio)                               │ │
│  │  • Primary change: Extend operating hours to 18-24 hours                           │ │
│  │  • Secondary: Enable walk-up transactions                                          │ │
│  │  • Estimated upgrade rate: 78%                                                     │ │
│  │  • Investment required: $XXX per site                                              │ │
│  │                                                                                    │ │
│  │  INTERVENTION B: "Screen Expansion"                                                │ │
│  │  ─────────────────────────────────                                                 │ │
│  │  • Applies to: 28 sites (23% of low-value portfolio)                               │ │
│  │  • Primary change: Add 1-2 screens                                                 │ │
│  │  • Secondary: Upgrade to video content                                             │ │
│  │  • Estimated upgrade rate: 71%                                                     │ │
│  │  • Investment required: $XXX per site                                              │ │
│  │                                                                                    │ │
│  │  INTERVENTION C: "Programmatic Enablement"                                         │ │
│  │  ─────────────────────────────────────────                                         │ │
│  │  • Applies to: 35 sites (28% of low-value portfolio)                               │ │
│  │  • Primary change: Enable Vistar programmatic                                      │ │
│  │  • Secondary: EMV/NFC payment support                                              │ │
│  │  • Estimated upgrade rate: 82%                                                     │ │
│  │  • Investment required: $XXX per site                                              │ │
│  │                                                                                    │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

```
╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                           IMPLEMENTATION ROADMAP                                         ║
╠═════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  PHASE 1: Foundation (Week 1-2)                                                          ║
║  ──────────────────────────────                                                          ║
║                                                                                          ║
║    □ Install dependencies: mapie, dice-ml                                                ║
║    □ Create site_scoring/explainability/ module structure                                ║
║    □ Implement ProbabilityCalibrator class                                               ║
║    □ Implement ConformalClassifier wrapper                                               ║
║    □ Unit tests for calibration and conformal prediction                                 ║
║                                                                                          ║
║    Deliverable: Uncertainty quantification working with existing model                   ║
║                                                                                          ║
║  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    ║
║                                                                                          ║
║  PHASE 2: Counterfactuals (Week 2-3)                                                     ║
║  ───────────────────────────────────                                                     ║
║                                                                                          ║
║    □ Define ACTIONABLE vs IMMUTABLE feature lists                                        ║
║    □ Implement CounterfactualGenerator class                                             ║
║    □ Define permitted_range constraints for monotonic features                           ║
║    □ Implement UpgradePathClusterer                                                      ║
║    □ Unit tests for counterfactual generation                                            ║
║                                                                                          ║
║    Deliverable: Single-site counterfactual explanations working                          ║
║                                                                                          ║
║  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    ║
║                                                                                          ║
║  PHASE 3: Integration (Week 3-4)                                                         ║
║  ───────────────────────────────                                                         ║
║                                                                                          ║
║    □ Implement TierClassifier with business labels                                       ║
║    □ Build ExplainabilityPipeline unified interface                                      ║
║    □ Integrate with training_service.py                                                  ║
║    □ Add calibration step to training flow                                               ║
║    □ Save/load pipeline components                                                       ║
║                                                                                          ║
║    Deliverable: Complete explain_site() pipeline                                         ║
║                                                                                          ║
║  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    ║
║                                                                                          ║
║  PHASE 4: UI/API (Week 4-5)                                                              ║
║  ──────────────────────────                                                              ║
║                                                                                          ║
║    □ Add /api/explain/{site_id} endpoint                                                 ║
║    □ Add /api/fleet-interventions endpoint                                               ║
║    □ Create explanation card component in UI                                             ║
║    □ Add counterfactual visualization                                                    ║
║    □ Add tier classification to site details panel                                       ║
║                                                                                          ║
║    Deliverable: Executive-ready explanations in web interface                            ║
║                                                                                          ║
║  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    ║
║                                                                                          ║
║  PHASE 5: Fleet Analysis (Week 5-6)                                                      ║
║  ──────────────────────────────────                                                      ║
║                                                                                          ║
║    □ Batch counterfactual generation (with progress tracking)                            ║
║    □ Clustering and intervention identification                                          ║
║    □ Strategic recommendation generation                                                 ║
║    □ Export to executive-friendly format (PDF/Excel)                                     ║
║                                                                                          ║
║    Deliverable: Fleet-wide strategic intervention report                                 ║
║                                                                                          ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## Dependencies to Add

```
# requirements.txt additions

# Conformal prediction
mapie>=0.8.0

# Counterfactual explanations
dice-ml>=0.11

# Already present - verify versions
shap>=0.44.0
scikit-learn>=1.3.0
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Conformal method | APS (Adaptive Prediction Sets) | Guarantees non-empty sets, better for business |
| Calibration method | Isotonic Regression | Non-parametric, handles complex calibration curves |
| Counterfactual method | DiCE genetic | Tree-agnostic, works with PyTorch wrapper |
| Clustering method | K-Means | Simple, interpretable cluster centers |
| Tier thresholds | 0.85 / 0.65 / 0.50 | Industry-standard for risk stratification |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| DiCE slow for large batches | High | Process in batches, cache results, use random method for exploration |
| MAPIE requires calibration data | Medium | Reserve 15% of validation for calibration (already planned) |
| sklearn CalibratedClassifierCV compatibility | Medium | Use manual IsotonicRegression (documented workaround) |
| Counterfactuals may suggest impossible changes | Medium | Strict permitted_range constraints, domain expert review |

---

## Success Metrics

1. **Coverage Guarantee**: Conformal prediction achieves ≥90% coverage on holdout
2. **Calibration**: Reliability diagram shows diagonal alignment (ECE < 0.05)
3. **Counterfactual Validity**: 95%+ of generated counterfactuals are actionable
4. **Business Adoption**: Executives use tier classification in decision-making

---

*Document generated for review. Awaiting approval before implementation.*
