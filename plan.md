# Implementation Plan: Standard Deviation Range Filter for Lookalike Classification

## Summary

Add a `lookalike_method` selector (`"percentile"` or `"std_dev"`) that lets users choose between percentile-based and standard-deviation-based revenue thresholds for binarizing the lookalike classification target. Mirrors the existing percentile implementation exactly — same config → service → data_loader flow, same UI pattern (inputs + hint + validation).

Also fixes existing bug: `start_training()` silently drops user-selected percentile values.

---

## Files Modified (5 files, ~11 surgical edits)

| File | Lines Affected | Change |
|------|----------------|--------|
| `site_scoring/config.py` | 46 | Add 3 fields after existing percentile fields |
| `src/services/training_service.py` | 118, 1247, 1827, 1967 | Mirror fields, fix bug, pass to pytorch, persist |
| `site_scoring/data_loader.py` | 224-253 | Branch `_process_target()` on method |
| `templates/index.html` | 3822-3836, 7059-7069, 7400-7407, 7410-7446, 7928-7959, 7968-7988 | UI: selector, inputs, hints, validation, config |

---

## Step-by-Step Implementation

### Step 1: `site_scoring/config.py` — Add 3 config fields

**Location**: After line 46 (`lookalike_upper_percentile: int = 100`)

**Add**:
```python
    # Lookalike classifier method: "percentile" or "std_dev"
    lookalike_method: str = "percentile"
    # Standard deviation bounds (used when method == "std_dev")
    lookalike_lower_std: float = 1.0   # σ above mean for lower threshold
    lookalike_upper_std: Optional[float] = None  # σ above mean for upper (None = no cap)
```

**Verification**: `python3 -c "from site_scoring.config import Config; c = Config(); print(c.lookalike_method, c.lookalike_lower_std, c.lookalike_upper_std)"`

---

### Step 2: `src/services/training_service.py` — Mirror 3 fields in TrainingConfig

**Location**: After line 118 (`lookalike_upper_percentile: int = 100`)

**Add**:
```python
    lookalike_method: str = "percentile"  # "percentile" or "std_dev"
    lookalike_lower_std: float = 1.0      # σ above mean (lower bound)
    lookalike_upper_std: Optional[float] = None  # σ above mean (upper bound, None = no cap)
```

**Also**: Ensure `Optional` is imported from `typing` (check existing imports).

**Verification**: `python3 -c "from src.services.training_service import TrainingConfig; c = TrainingConfig(); print(c.lookalike_method)"`

---

### Step 3: `src/services/training_service.py` — Fix bug + add extraction in `start_training()`

**Location**: Lines 1827-1828 (inside `TrainingConfig(...)` constructor, after `cluster_probability_threshold`)

**Add** (fixes the missing percentile extraction AND adds new fields):
```python
            # Lookalike classifier configuration
            lookalike_lower_percentile=int(config_dict.get("lookalike_lower_percentile", 90)),
            lookalike_upper_percentile=int(config_dict.get("lookalike_upper_percentile", 100)),
            lookalike_method=config_dict.get("lookalike_method", "percentile"),
            lookalike_lower_std=float(config_dict.get("lookalike_lower_std", 1.0)),
            lookalike_upper_std=float(config_dict.get("lookalike_upper_std")) if config_dict.get("lookalike_upper_std") is not None else None,
```

**Verification**: Confirmed by tracing that `config_dict` comes directly from `request.get_json()` in `src/routes/training.py:123`.

---

### Step 4: `src/services/training_service.py` — Pass new fields in `run_training_logic()`

**Location**: After line 1247 (`pytorch_config.lookalike_upper_percentile = ...`)

**Add**:
```python
    pytorch_config.lookalike_method = config.lookalike_method
    pytorch_config.lookalike_lower_std = config.lookalike_lower_std
    pytorch_config.lookalike_upper_std = config.lookalike_upper_std
```

**Verification**: Fields flow from `TrainingConfig` → `Config` (pytorch) → `DataProcessor._process_target()`.

---

### Step 5: `src/services/training_service.py` — Persist in experiment catalog

**Location**: After line 1967 (`"lookalike_upper_percentile": ...`)

**Add**:
```python
        "lookalike_method": config.get("lookalike_method", "percentile"),
        "lookalike_lower_std": config.get("lookalike_lower_std"),
        "lookalike_upper_std": config.get("lookalike_upper_std"),
```

**Verification**: `GET /api/experiments/catalog` will include the new fields for each experiment.

---

### Step 6: `site_scoring/data_loader.py` — Branch `_process_target()` on method

**Location**: Lines 224-253 (the `if self.config.task_type == "lookalike":` block)

**Replace** the threshold calculation section (lines 224-253) with:

```python
        if self.config.task_type == "lookalike":
            method = getattr(self.config, 'lookalike_method', 'percentile')

            if method == "std_dev":
                # Standard deviation binarization
                lower_std = getattr(self.config, 'lookalike_lower_std', 1.0)
                upper_std = getattr(self.config, 'lookalike_upper_std', None)

                mean_val = float(np.mean(target_data))
                std_val = float(np.std(target_data))

                lower_threshold = mean_val + (lower_std * std_val)
                upper_threshold = mean_val + (upper_std * std_val) if upper_std is not None else float('inf')

                self.top_performer_threshold = lower_threshold
                self.top_performer_upper_threshold = upper_threshold

                if upper_std is None:
                    binary_labels = (target_data >= lower_threshold).astype(np.float32)
                else:
                    binary_labels = ((target_data >= lower_threshold) & (target_data <= upper_threshold)).astype(np.float32)

                n_positive = int(binary_labels.sum())
                pct_positive = n_positive / len(binary_labels) * 100
                print(f"Classification: std_dev method (mean={mean_val:,.0f}, std={std_val:,.0f})")
                print(f"  Lower threshold: ${lower_threshold:,.0f} (mean + {lower_std}σ)")
                if upper_std is not None:
                    print(f"  Upper threshold: ${upper_threshold:,.0f} (mean + {upper_std}σ)")
                print(f"  Top performers: {n_positive}/{len(binary_labels)} sites ({pct_positive:.1f}%)")
            else:
                # Percentile binarization (existing logic, unchanged)
                lower_pct = getattr(self.config, 'lookalike_lower_percentile', 90)
                upper_pct = getattr(self.config, 'lookalike_upper_percentile', 100)

                lower_threshold = float(np.percentile(target_data, lower_pct))
                upper_threshold = float(np.percentile(target_data, upper_pct)) if upper_pct < 100 else float('inf')

                self.top_performer_threshold = lower_threshold
                self.top_performer_upper_threshold = upper_threshold

                if upper_pct >= 100:
                    binary_labels = (target_data >= lower_threshold).astype(np.float32)
                else:
                    binary_labels = ((target_data >= lower_threshold) & (target_data <= upper_threshold)).astype(np.float32)

                n_positive = int(binary_labels.sum())
                pct_positive = n_positive / len(binary_labels) * 100
                print(f"Classification: revenue range p{lower_pct}-p{upper_pct}")
                print(f"  Lower threshold: ${lower_threshold:,.0f} (p{lower_pct})")
                if upper_pct < 100:
                    print(f"  Upper threshold: ${upper_threshold:,.0f} (p{upper_pct})")
                print(f"  Top performers: {n_positive}/{len(binary_labels)} sites ({pct_positive:.1f}%)")

            self.target_scaler = None
            return torch.from_numpy(np.ascontiguousarray(binary_labels, dtype=np.float32))
```

**Key**: The `self.target_scaler = None` and `return` lines are shared by both branches (moved outside the if/else).

**Verification**: `python3 -c "from site_scoring.config import Config; c = Config(task_type='lookalike', lookalike_method='std_dev', lookalike_lower_std=1.5); print('Config OK:', c.lookalike_method)"`

---

### Step 7: `templates/index.html` — Add method selector and std_dev inputs to HTML

**Location**: Inside the `#lookalike-percentile-config` div (line 3822), add a method selector BEFORE the existing percentile inputs row.

**Add** (after the opening `<div id="lookalike-percentile-config"` line, before the first `training-form-row`):

```html
    <div class="training-form-row">
        <div class="training-form-group" style="flex: 1;">
            <label class="training-form-label" data-tooltip="Method for defining 'top performer' threshold. Percentile selects top N% of sites. Std Dev selects sites N standard deviations above mean revenue.">Classification Method</label>
            <select class="training-form-input" id="lookalike-method">
                <option value="percentile" selected>Percentile</option>
                <option value="std_dev">Standard Deviation</option>
            </select>
        </div>
    </div>
```

**Then** add a NEW row for std_dev inputs AFTER the existing percentile row (after the `</div>` closing the percentile `training-form-row`):

```html
    <div class="training-form-row" id="stddev-inputs-row" style="display: none;">
        <div class="training-form-group">
            <label class="training-form-label" data-tooltip="Lower bound in standard deviations above the mean. Sites with revenue ≥ mean + Nσ are 'top performers'. Typical: 1.0σ ≈ top 16%, 1.5σ ≈ top 7%, 2.0σ ≈ top 2%.">Lower σ</label>
            <input type="number" class="training-form-input" id="lookalike-lower-std" value="1.0" min="0" max="4" step="0.1">
        </div>
        <div class="training-form-group">
            <label class="training-form-label" data-tooltip="Upper bound in standard deviations (optional). Leave blank or set high to include all sites above lower bound. Set a value to select a band (e.g., between 1σ and 2σ).">Upper σ</label>
            <input type="number" class="training-form-input" id="lookalike-upper-std" value="" min="0" max="5" step="0.1" placeholder="No cap">
        </div>
    </div>
```

**Rename** the existing percentile row div: wrap it with `id="percentile-inputs-row"` for toggle control.

---

### Step 8: `templates/index.html` — Add event listeners

**Location**: After the existing percentile event listeners (~line 7069)

**Add**:
```javascript
    // Lookalike method selector
    const lookalikeMethodSelect = document.getElementById('lookalike-method');
    if (lookalikeMethodSelect) {
        lookalikeMethodSelect.addEventListener('change', function() {
            const isStdDev = this.value === 'std_dev';
            document.getElementById('percentile-inputs-row').style.display = isStdDev ? 'none' : '';
            document.getElementById('stddev-inputs-row').style.display = isStdDev ? '' : 'none';
            updateLookalikeHint();
        });
    }
    // Std dev input listeners
    const lowerStdInput = document.getElementById('lookalike-lower-std');
    const upperStdInput = document.getElementById('lookalike-upper-std');
    if (lowerStdInput) {
        lowerStdInput.addEventListener('input', updateLookalikeHint);
        lowerStdInput.addEventListener('change', updateLookalikeHint);
    }
    if (upperStdInput) {
        upperStdInput.addEventListener('input', updateLookalikeHint);
        upperStdInput.addEventListener('change', updateLookalikeHint);
    }
```

---

### Step 9: `templates/index.html` — Rename and extend hint function

**Location**: Replace `updatePercentileHint()` (~lines 7410-7446) entirely.

**Rename to** `updateLookalikeHint()` and handle both modes:

```javascript
function updateLookalikeHint() {
    const method = document.getElementById('lookalike-method')?.value || 'percentile';
    const hintEl = document.getElementById('percentile-hint');
    const errorEl = document.getElementById('percentile-error');
    if (!hintEl || !errorEl) return;

    if (method === 'std_dev') {
        const lowerStd = parseFloat(document.getElementById('lookalike-lower-std')?.value);
        const upperStdVal = document.getElementById('lookalike-upper-std')?.value;
        const upperStd = upperStdVal ? parseFloat(upperStdVal) : null;

        let error = '';
        if (isNaN(lowerStd) || lowerStd < 0 || lowerStd > 4) {
            error = 'Lower σ must be between 0 and 4';
        } else if (upperStd !== null && (isNaN(upperStd) || upperStd <= lowerStd)) {
            error = 'Upper σ must be greater than lower σ';
        }

        if (error) {
            errorEl.textContent = error;
            errorEl.style.display = 'block';
            hintEl.style.display = 'none';
        } else {
            errorEl.style.display = 'none';
            hintEl.style.display = 'block';
            if (upperStd === null) {
                hintEl.textContent = `Sites ≥ mean + ${lowerStd}σ revenue will be labeled as top performers`;
            } else {
                hintEl.textContent = `Sites between mean + ${lowerStd}σ and mean + ${upperStd}σ revenue will be labeled as top performers`;
            }
        }
    } else {
        // Original percentile logic (unchanged)
        const lower = parseInt(document.getElementById('lookalike-lower-percentile')?.value, 10);
        const upper = parseInt(document.getElementById('lookalike-upper-percentile')?.value, 10);

        let error = '';
        if (isNaN(lower) || lower < 1 || lower > 99) {
            error = 'Lower percentile must be an integer between 1 and 99';
        } else if (isNaN(upper) || upper < 1 || upper > 100) {
            error = 'Upper percentile must be an integer between 1 and 100';
        } else if (upper <= lower) {
            error = 'Upper percentile must be greater than lower percentile';
        }

        if (error) {
            errorEl.textContent = error;
            errorEl.style.display = 'block';
            hintEl.style.display = 'none';
        } else {
            errorEl.style.display = 'none';
            hintEl.style.display = 'block';
            const pctSites = upper >= 100 ? (100 - lower) : (upper - lower);
            if (upper >= 100) {
                hintEl.textContent = `Top ${100 - lower}% of sites by revenue will be labeled as top performers`;
            } else {
                hintEl.textContent = `Sites between p${lower} and p${upper} (${pctSites}% of sites) will be labeled as top performers`;
            }
        }
    }
}
```

**Also**: Update ALL references from `updatePercentileHint` → `updateLookalikeHint`:
- Event listeners at ~7063, 7065, 7068 (existing percentile listeners)
- `handleTrainingTaskChange()` at ~7406

---

### Step 10: `templates/index.html` — Update `getTrainingConfig()`

**Location**: After the existing percentile extraction (~lines 7928-7929, 7957-7959)

**Add** to variable extraction section:
```javascript
const lookalikeMethod = document.getElementById('lookalike-method')?.value || 'percentile';
const lowerStd = parseFloat(document.getElementById('lookalike-lower-std')?.value || '1.0');
const upperStdVal = document.getElementById('lookalike-upper-std')?.value;
const upperStd = upperStdVal ? parseFloat(upperStdVal) : null;
```

**Add** to returned config object:
```javascript
    lookalike_method: lookalikeMethod,
    lookalike_lower_std: lowerStd,
    lookalike_upper_std: upperStd,
```

---

### Step 11: `templates/index.html` — Update `validatePercentileBounds()` → `validateLookalikeBounds()`

**Location**: Replace the existing function (~lines 7968-7988)

**New function**:
```javascript
function validateLookalikeBounds() {
    const taskType = document.getElementById('task-type').value;
    if (taskType !== 'lookalike') return { valid: true };

    const method = document.getElementById('lookalike-method')?.value || 'percentile';

    if (method === 'std_dev') {
        const lowerStd = parseFloat(document.getElementById('lookalike-lower-std')?.value);
        const upperStdVal = document.getElementById('lookalike-upper-std')?.value;
        const upperStd = upperStdVal ? parseFloat(upperStdVal) : null;

        if (isNaN(lowerStd) || lowerStd < 0 || lowerStd > 4) {
            return { valid: false, error: 'Lower σ must be between 0 and 4' };
        }
        if (upperStd !== null && (isNaN(upperStd) || upperStd <= lowerStd)) {
            return { valid: false, error: 'Upper σ must be greater than lower σ' };
        }
        return { valid: true };
    } else {
        // Existing percentile validation
        const lower = parseInt(document.getElementById('lookalike-lower-percentile')?.value || '90', 10);
        const upper = parseInt(document.getElementById('lookalike-upper-percentile')?.value || '100', 10);

        if (isNaN(lower) || lower < 1 || lower > 99) {
            return { valid: false, error: 'Lower percentile must be an integer between 1 and 99' };
        }
        if (isNaN(upper) || upper < 1 || upper > 100) {
            return { valid: false, error: 'Upper percentile must be an integer between 1 and 100' };
        }
        if (upper <= lower) {
            return { valid: false, error: 'Upper percentile must be greater than lower percentile' };
        }
        return { valid: true };
    }
}
```

**Also**: Update `startTraining()` call from `validatePercentileBounds()` → `validateLookalikeBounds()` (~line 7995).

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Existing percentile flow breaks | High | Percentile is the default; all existing code paths unchanged when `method == "percentile"` |
| `start_training()` bug fix changes default behavior | Low | Bug fix makes UI-selected values actually work — defaults remain 90/100 |
| `upper_std = None` serialization to JSON | Medium | Python `None` → JSON `null` → JS `null`. Handle in `getTrainingConfig()` with ternary. |
| `std_dev` with skewed revenue selects unexpected count | Low | Hint text warns user; actual count shown in training progress output |

## Rollback

All changes are additive. To rollback:
1. Remove 3 new fields from `config.py` and `training_service.py`
2. Revert `_process_target()` to original (just the percentile branch)
3. Remove new HTML elements and JS functions
4. Revert function renames (`updateLookalikeHint` → `updatePercentileHint`, etc.)

No database migrations, no schema changes, no external dependencies.

## Verification

```bash
# 1. Unit: import check
python3 -c "from site_scoring.config import Config; c = Config(lookalike_method='std_dev', lookalike_lower_std=1.5); print(c)"

# 2. Run existing tests (no regression)
pytest tests/ -v --ignore=tests/slow -x

# 3. Start app and test UI
python3 app.py  # → http://localhost:8080
# → Select "Lookalike" task → verify method dropdown appears
# → Switch to "Standard Deviation" → verify percentile inputs hide, std inputs show
# → Train with std_dev method → verify training log shows "std_dev method" in console
```
