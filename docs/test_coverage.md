# Test Coverage — src/services/

This document describes the test coverage for the `src/services/` directory, including what is tested, what edge cases are covered, and how to run the tests.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests (fast, no real data needed)
pytest tests/test_data_service_unit.py tests/test_training_service_unit.py tests/test_geospatial_services_unit.py -v

# Run integration tests (requires real CSV data)
pytest tests/test_data_loading.py tests/test_revenue_consistency.py -v

# Run with coverage report
pytest tests/ --cov=src/services --cov-report=term-missing
```

---

## Test File Organization

```
tests/
├── conftest.py                        # Shared fixtures (Flask client, sample data)
│
│── Unit Tests (mocked, fast, no real data)
├── test_data_service_unit.py          # data_service.py logic and edge cases
├── test_training_service_unit.py      # training_service.py logic and edge cases
├── test_geospatial_services_unit.py   # interstate_distance.py + nearest_site.py
│
│── Integration Tests (requires real CSV/data files)
├── test_data_loading.py               # Data loading with real files
├── test_revenue_consistency.py        # Revenue pipeline consistency
├── test_api_sites.py                  # Flask API endpoint tests
├── test_api_training.py               # Training API endpoint tests
├── test_frontend_data.py              # Frontend data completeness
└── test_regression.py                 # ML model output validation
```

---

## Coverage Summary

| Service File | Function | Unit Tests | Integration Tests | Edge Cases |
|-------------|----------|:----------:|:-----------------:|:----------:|
| **data_service.py** | | | | |
| | `_clean_nan_values()` | ✅ 22 tests | — | NaN, Inf, -Inf, numpy types, nested, arrays |
| | `load_sites()` | ✅ 4 tests | ✅ 7 tests | Missing coords, duplicates, column rename |
| | `load_revenue_metrics()` | ✅ 3 tests | ✅ 5 tests | Zero months, null revenue, score clamping |
| | `load_site_details()` | — | ✅ 5 tests | — |
| | `get_filter_options()` | ✅ 3 tests | ✅ 4 tests | Empty strings, NaN, sorting |
| | `get_filtered_site_ids()` | ✅ 7 tests | — | Empty filters, invalid fields, None/empty values |
| | `get_site_details_for_display()` | ✅ 8 tests | — | Unknown site, partial data, NaN cleaning |
| | `preload_all_data()` | — | ✅ 2 tests | — |
| **interstate_distance.py** | | | | |
| | `distance_to_nearest_interstate()` | ✅ 6 tests | — | Nearest point, highway segment, non-negative |
| | `batch_distance_to_interstate()` | ✅ 8 tests | — | Custom columns, row count, distance derivation |
| | `preload_highway_data()` | ✅ 1 test | — | Calls both load and index |
| | Constants | ✅ 2 tests | — | METERS_PER_MILE, TIGER_URL |
| **nearest_site.py** | | | | |
| | `calculate_nearest_site_distances()` | ✅ 10 tests | — | Self-match, nearby/distant, 2-site, unnamed cols |
| **training_service.py** | | | | |
| | `_sanitize_for_json()` | ✅ 11 tests | — | Inf, NaN, nested, JSON serializable |
| | `detect_apple_chip()` | ✅ 6 tests | — | M1/M4 Pro/Ultra, Intel, subprocess failure |
| | `get_optimized_training_params()` | ✅ 10 tests | — | All chips, batch capping, worker/prefetch scaling |
| | `get_system_info()` | ✅ 2 tests | ✅ 4 tests | Device availability |
| | `start_training()` | — | ✅ 3 tests | Duplicate job prevention |
| | `stop_training()` | ✅ 1 test | ✅ 2 tests | No job exists |
| | `get_training_status()` | ✅ 1 test | ✅ 1 test | No job exists |
| | `stream_training_progress()` | — | ✅ 3 tests | SSE format, headers |
| | `TrainingConfig` | ✅ 6 tests | — | Defaults, overrides, device detection |
| | `TrainingProgress` | ✅ 3 tests | — | Defaults, required fields |
| | `APPLE_CHIP_SPECS` | ✅ 4 tests | — | Required keys, monotonic tiers/cores/batch |
| **epa_walkability.py** | | | | |
| | All functions | — | — | Not tested (external data dependency) |

---

## Detailed Test Descriptions

### test_data_service_unit.py

#### TestCleanNanValues (22 tests)
Tests the JSON sanitization function that prevents `NaN`/`Inf` values from breaking API responses.

| Test | Edge Case | Why It Matters |
|------|-----------|----------------|
| `test_nan_becomes_none` | `float('nan')` | JSON spec doesn't support NaN |
| `test_inf_becomes_none` | `float('inf')` | JSON spec doesn't support Infinity |
| `test_negative_inf_becomes_none` | `float('-inf')` | Negative infinity same issue |
| `test_numpy_float64_nan_becomes_none` | `np.float64('nan')` | Pandas returns numpy types |
| `test_numpy_int64_converted` | `np.int64(42)` | Must be Python int for JSON |
| `test_numpy_array_converted` | `np.array([1, nan, 3])` | Arrays need recursive handling |
| `test_deeply_nested_structure` | `{a: {b: [{val: nan}]}}` | Real API responses are nested |
| `test_zero_float_not_cleaned` | `0.0` | Zero is valid, don't null it |
| `test_mixed_numpy_and_python_types` | All types in one dict | Real-world data mixing |

#### TestGetFilteredSiteIds (7 tests)
Tests the site filtering logic used by the filter panel.

| Test | Edge Case | Why It Matters |
|------|-----------|----------------|
| `test_empty_filters_returns_empty` | `{}` | UI sends empty on clear |
| `test_multiple_filters_intersection` | State + Network | AND logic between filters |
| `test_invalid_field_name_ignored` | Unknown display name | UI could send stale fields |
| `test_none_value_filter_skipped` | `{"State": None}` | Unset dropdown value |
| `test_empty_string_filter_skipped` | `{"State": ""}` | Cleared dropdown |

#### TestGetSiteDetailsForDisplay (8 tests)
Tests the comprehensive site detail builder.

| Test | Edge Case | Why It Matters |
|------|-----------|----------------|
| `test_returns_none_for_unknown_site` | Nonexistent GTVID | 404 handling |
| `test_partial_data_for_coords_only_site` | Has coords, no details | Historical/new sites |
| `test_site_with_revenue_but_no_coords_or_details` | Revenue-only site | Data pipeline gaps |
| `test_nan_values_cleaned_in_output` | NaN in detail fields | Prevents JSON errors |
| `test_categories_structure` | All 8 categories present | Frontend expects all |

#### TestLoadSitesEdgeCases (4 tests)

| Test | Edge Case | Why It Matters |
|------|-----------|----------------|
| `test_drops_rows_with_missing_lat` | `None` latitude | Can't plot on map |
| `test_deduplicates_by_gtvid` | Same GTVID twice | Monthly data has repeats |
| `test_columns_renamed_correctly` | lowercase → CamelCase | Frontend expects GTVID |

#### TestLoadRevenueMetricsEdgeCases (3 tests)

| Test | Edge Case | Why It Matters |
|------|-----------|----------------|
| `test_score_clamped_to_zero_one` | Very high/low revenue | Color gradient breaks |
| `test_handles_zero_active_months` | 0 months active | Division by zero |
| `test_drops_null_revenue_rows` | `None` revenue values | Aggregation accuracy |

---

### test_training_service_unit.py

#### TestSanitizeForJson (11 tests)
Tests the training-specific JSON sanitizer (used before SSE streaming).

| Test | Edge Case | Why It Matters |
|------|-----------|----------------|
| `test_mixed_types_in_dict` | Progress-like payload | Real training updates |
| `test_result_is_json_serializable` | Full sanitized output | SSE would crash otherwise |
| `test_zero_not_sanitized` | `0.0` loss value | Early epochs have zero loss |

#### TestGetOptimizedTrainingParams (10 tests)
Tests the Apple Silicon optimization logic.

| Test | Edge Case | Why It Matters |
|------|-----------|----------------|
| `test_batch_size_capped_by_chip` | Request > chip max | OOM prevention |
| `test_batch_size_not_increased` | Request < chip max | User intent preserved |
| `test_unknown_chip_uses_defaults` | Unrecognized chip ID | Future-proofing |
| `test_workers_capped_at_eight` | High-tier chip | Diminishing returns > 8 |
| `test_all_known_chips_valid` | Every entry in SPECS | No missing keys |

#### TestDetectAppleChip (6 tests)
Tests system hardware detection with mocked subprocess calls.

| Test | Edge Case | Why It Matters |
|------|-----------|----------------|
| `test_detects_m4_pro` | "Apple M4 Pro" string | Current hardware |
| `test_detects_m2_ultra` | "Apple M2 Ultra" variant | High-end workstations |
| `test_handles_non_apple_cpu` | Intel CPU string | CI/CD environments |
| `test_handles_subprocess_failure` | Exception thrown | Docker/sandboxed envs |

#### TestTrainingConfig (6 tests)
Tests the training configuration dataclass defaults.

#### TestTrainingProgress (3 tests)
Tests the progress update data container.

#### TestTrainingLifecycle (4 tests)
Tests start/stop/status without actually running training.

#### TestAppleChipSpecs (4 tests)
Validates the chip specification lookup table structure.

---

### test_geospatial_services_unit.py

#### TestDistanceToNearestInterstate (6 tests)
Tests single-point Interstate distance queries with mocked spatial data.

| Test | Edge Case | Why It Matters |
|------|-----------|----------------|
| `test_distance_is_non_negative` | Any coordinate | Physical distance ≥ 0 |
| `test_miles_equals_meters_divided_by_constant` | Conversion accuracy | Feature consistency |
| `test_include_nearest_point_adds_coords` | Optional param | Highway connection viz |
| `test_include_highway_segment_adds_coords` | Optional param | Map line drawing |

#### TestBatchDistanceToInterstate (8 tests)
Tests bulk distance calculations for the data pipeline.

| Test | Edge Case | Why It Matters |
|------|-----------|----------------|
| `test_preserves_original_columns` | Input columns kept | No data loss |
| `test_same_row_count_as_input` | Output size match | No duplication/loss |
| `test_custom_column_names` | Non-default col names | Script compatibility |
| `test_all_distances_non_negative` | Physical constraint | Model feature validity |

#### TestCalculateNearestSiteDistances (10 tests)
Tests KDTree-based nearest neighbor calculations.

| Test | Edge Case | Why It Matters |
|------|-----------|----------------|
| `test_nearest_site_is_not_self` | Self-exclusion | K=3 query handles this |
| `test_nearby_sites_have_small_distances` | Geographic proximity | Algorithm correctness |
| `test_distant_sites_have_large_distances` | LA vs NYC | Not random matching |
| `test_two_sites_are_each_others_nearest` | Minimum viable input | Boundary condition |
| `test_handles_unnamed_columns` | Excel CSV exports | Data pipeline robustness |

---

## Not Tested (and Why)

### epa_walkability.py
**Reason**: Requires downloading ~220MB EPA Smart Location Database and Census TIGER/Line shapefiles. The function performs spatial joins against block group polygons which cannot be meaningfully mocked without reproducing the entire geospatial logic.

**Mitigation**: The walkability service is not called by the running application (`app.py`) — it's only used by `scripts/site_walkability.py` which is a standalone batch tool.

### TrainingJob._run_training() (full training loop)
**Reason**: Requires loading the full training dataset (~26K sites × 94 features) and running PyTorch model training. Even with `epochs=1`, this takes significant time and memory.

**Mitigation**: Tested indirectly via `test_api_training.py` integration tests which exercise the full training lifecycle through the API.

### _load_highways() (real shapefile download)
**Reason**: Downloads ~15MB shapefile from Census Bureau on first run. Network dependency makes it unsuitable for unit tests.

**Mitigation**: All functions that use `_load_highways()` are tested with mocked highway data via `@patch`.

---

## Edge Case Categories

### 1. Null/NaN Handling
- `_clean_nan_values()`: Python NaN, numpy NaN, Inf, -Inf, np.floating, np.integer, np.ndarray
- `_sanitize_for_json()`: Same patterns for training progress
- `load_sites()`: Missing lat/lon coordinates dropped
- `load_revenue_metrics()`: Null revenue rows excluded
- `get_filter_options()`: NaN values excluded from dropdown options
- `get_site_details_for_display()`: NaN cleaned before JSON response

### 2. Division by Zero
- `load_revenue_metrics()`: `active_months.clip(lower=1)` prevents /0
- Revenue score normalization: `p90 > p10` guard prevents /0

### 3. Boundary Conditions
- `get_filtered_site_ids()`: Empty filters, None values, empty strings
- `calculate_nearest_site_distances()`: 2 sites only, self-exclusion
- `get_optimized_training_params()`: Batch size at exact chip maximum
- Revenue score clamping: `max(0, min(1, score))`

### 4. Type Coercion
- `_clean_nan_values()`: np.int64 → int, np.float64 → float, np.ndarray → list
- All JSON serialization paths must produce native Python types

### 5. External System Failures
- `detect_apple_chip()`: subprocess failure returns defaults
- Unknown chip IDs use fallback parameters
- Non-Apple CPUs (Intel) handled gracefully

### 6. Data Pipeline Integrity
- `batch_distance_to_interstate()`: Same row count as input
- `calculate_nearest_site_distances()`: No self-matching
- `load_sites()`: Deduplication by GTVID
- Revenue totals match between raw aggregation and service layer

---

## Test Dependencies

### Unit Tests (no external dependencies)
```
pytest
numpy
pandas
unittest.mock
```

### Integration Tests (require data files)
```
data/input/Site Scores - Site Revenue, Impressions, and Diagnostics.csv
data/input/nearest_site_distances.csv
data/input/site_interstate_distances.csv
data/processed/site_training_data.parquet
```

### Geospatial Unit Tests (require libraries, not data)
```
geopandas
shapely
scipy
```

---

## Adding New Tests

When adding new functionality to `src/services/`, follow this pattern:

1. **Unit test** in `test_*_unit.py` with mocked dependencies
2. **Integration test** in the appropriate integration file if it touches real data
3. **Edge cases** to consider:
   - What if the input is None/NaN/empty?
   - What if the input has zero elements?
   - What if the input has exactly one element?
   - What if the result would be Inf or NaN?
   - What if a subprocess/network call fails?
   - Can the output be JSON-serialized?
