/**
 * Training Configuration — Feature selection, config gathering, validation.
 * Used on the home page for experiment setup.
 *
 * Depends on: shared-utils.js (getFeatureDisplayName, APPLE_CHIP_SPECS)
 */

// ============================================================================
// Feature Selection State
// ============================================================================

let featureSelectionState = {
    numeric: new Set(),
    categorical: new Set(),
    boolean: new Set(),
    allFeatures: { numeric: [], categorical: [], boolean: [] }
};

// Current active tab in feature selector
let activeFeatureTab = 'numeric';

// ============================================================================
// Feature Selection Persistence (Session Storage)
// ============================================================================

function saveFeatureSelectionToSession() {
    const state = {
        numeric: Array.from(featureSelectionState.numeric),
        categorical: Array.from(featureSelectionState.categorical),
        boolean: Array.from(featureSelectionState.boolean),
    };
    sessionStorage.setItem('featureSelection', JSON.stringify(state));
}

function loadFeatureSelectionFromSession() {
    const saved = sessionStorage.getItem('featureSelection');
    if (saved) {
        try {
            const state = JSON.parse(saved);
            return {
                numeric: new Set(state.numeric || []),
                categorical: new Set(state.categorical || []),
                boolean: new Set(state.boolean || []),
            };
        } catch (e) {
            console.warn('Failed to parse saved feature selection:', e);
        }
    }
    return null;
}

// ============================================================================
// Feature Count Updates
// ============================================================================

function updateFeatureSelectionCounts() {
    const selectedCount = featureSelectionState.numeric.size +
                          featureSelectionState.categorical.size +
                          featureSelectionState.boolean.size;
    const totalCount = featureSelectionState.allFeatures.numeric.length +
                       featureSelectionState.allFeatures.categorical.length +
                       featureSelectionState.allFeatures.boolean.length;

    const countEl = document.getElementById('feature-selected-count');
    const totalEl = document.getElementById('feature-total-count');
    if (countEl) countEl.textContent = selectedCount;
    if (totalEl) totalEl.textContent = totalCount;

    // Update tab counts
    ['numeric', 'categorical', 'boolean'].forEach(type => {
        const tabCount = document.getElementById(`tab-count-${type}`);
        if (tabCount) {
            tabCount.textContent = `${featureSelectionState[type].size}/${featureSelectionState.allFeatures[type].length}`;
        }
    });

    // Warning if too few selected
    const warningEl = document.getElementById('feature-warning');
    if (warningEl) {
        warningEl.classList.toggle('show', selectedCount < 3);
    }

    saveFeatureSelectionToSession();
}

// ============================================================================
// Feature Checkbox Handlers
// ============================================================================

function handleFeatureCheckboxChange(feature, type, checked) {
    if (checked) {
        featureSelectionState[type].add(feature);
    } else {
        featureSelectionState[type].delete(feature);
    }

    const item = document.querySelector(`[data-feature="${feature}"]`);
    if (item) item.classList.toggle('unselected', !checked);

    updateFeatureSelectionCounts();
}

function selectAllFeatures(type) {
    featureSelectionState.allFeatures[type].forEach(f => {
        featureSelectionState[type].add(f);
        const cb = document.getElementById(`feature-${f}`);
        if (cb) cb.checked = true;
        const item = document.querySelector(`[data-feature="${f}"]`);
        if (item) item.classList.remove('unselected');
    });
    updateFeatureSelectionCounts();
}

function deselectAllFeatures(type) {
    featureSelectionState[type].clear();
    featureSelectionState.allFeatures[type].forEach(f => {
        const cb = document.getElementById(`feature-${f}`);
        if (cb) cb.checked = false;
        const item = document.querySelector(`[data-feature="${f}"]`);
        if (item) item.classList.add('unselected');
    });
    updateFeatureSelectionCounts();
}

function getSelectedFeatures() {
    return [
        ...featureSelectionState.numeric,
        ...featureSelectionState.categorical,
        ...featureSelectionState.boolean,
    ];
}

// ============================================================================
// Feature Tab Switching
// ============================================================================

function switchFeatureTab(type) {
    activeFeatureTab = type;

    // Update tab active state
    document.querySelectorAll('.feature-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.type === type);
    });

    // Show/hide grids
    document.querySelectorAll('.feature-grid').forEach(grid => {
        grid.style.display = grid.dataset.type === type ? 'grid' : 'none';
    });
}

// ============================================================================
// Feature Search
// ============================================================================

function filterFeatures(query) {
    const q = query.toLowerCase();
    document.querySelectorAll('.feature-item').forEach(item => {
        const name = (item.querySelector('label')?.textContent || '').toLowerCase();
        const raw = item.dataset.feature || '';
        const matches = !q || name.includes(q) || raw.includes(q);
        item.style.display = matches ? '' : 'none';
    });
}

// ============================================================================
// Load Features from API
// ============================================================================

async function loadTrainingFeatures() {
    const container = document.getElementById('feature-grids-container');
    if (!container) return;

    try {
        const response = await fetch('/api/training/all-features');
        const data = await response.json();

        featureSelectionState.allFeatures = {
            numeric: data.numeric || [],
            categorical: data.categorical || [],
            boolean: data.boolean || [],
        };

        // Restore from session, else select all
        const saved = loadFeatureSelectionFromSession();
        if (saved) {
            featureSelectionState.numeric = new Set(
                [...saved.numeric].filter(f => data.numeric.includes(f))
            );
            featureSelectionState.categorical = new Set(
                [...saved.categorical].filter(f => data.categorical.includes(f))
            );
            featureSelectionState.boolean = new Set(
                [...saved.boolean].filter(f => data.boolean.includes(f))
            );
        } else {
            featureSelectionState.numeric = new Set(data.numeric);
            featureSelectionState.categorical = new Set(data.categorical);
            featureSelectionState.boolean = new Set(data.boolean);
        }

        // Build grid HTML for each type
        function buildFeatureGrid(features, type) {
            const sorted = [...features].sort((a, b) => {
                return getFeatureDisplayName(a).toLowerCase()
                    .localeCompare(getFeatureDisplayName(b).toLowerCase());
            });

            return sorted.map(f => {
                const isSelected = featureSelectionState[type].has(f);
                const unselected = isSelected ? '' : ' unselected';
                return `<div class="feature-item${unselected}" data-feature="${f}" data-type="${type}">
                    <input type="checkbox" id="feature-${f}" ${isSelected ? 'checked' : ''}
                           onchange="handleFeatureCheckboxChange('${f}', '${type}', this.checked)">
                    <label for="feature-${f}" title="${f}">${getFeatureDisplayName(f)}</label>
                </div>`;
            }).join('');
        }

        container.innerHTML = `
            <div class="feature-grid" data-type="numeric" style="display: grid;">
                ${buildFeatureGrid(data.numeric, 'numeric')}
            </div>
            <div class="feature-grid" data-type="categorical" style="display: none;">
                ${buildFeatureGrid(data.categorical, 'categorical')}
            </div>
            <div class="feature-grid" data-type="boolean" style="display: none;">
                ${buildFeatureGrid(data.boolean, 'boolean')}
            </div>
        `;

        updateFeatureSelectionCounts();

    } catch (error) {
        console.error('Failed to load training features:', error);
        container.innerHTML = '<div style="font-size: 0.78rem; color: var(--text-secondary); padding: 12px;">Unable to load features</div>';
    }
}

// ============================================================================
// Task/Model Card Selection
// ============================================================================

let currentTaskType = 'regression';
let currentModelType = 'xgboost';

function selectTask(taskType) {
    currentTaskType = taskType;

    // Update card selection visuals
    document.querySelectorAll('.task-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.task === taskType);
    });

    // Auto-force model type for certain tasks
    if (taskType === 'lookalike' || taskType === 'clustering') {
        selectModel('neural_network');
        // Disable XGBoost card for these tasks
        const xgbCard = document.querySelector('.model-card[data-model="xgboost"]');
        if (xgbCard) xgbCard.classList.add('disabled');
    } else {
        const xgbCard = document.querySelector('.model-card[data-model="xgboost"]');
        if (xgbCard) xgbCard.classList.remove('disabled');
    }

    // Show/hide conditional sections
    const percentileConfig = document.getElementById('percentile-config');
    const clusterConfig = document.getElementById('cluster-config-section');
    if (percentileConfig) percentileConfig.style.display = taskType === 'lookalike' ? 'block' : 'none';
    if (clusterConfig) clusterConfig.style.display = taskType === 'clustering' ? 'block' : 'none';

    // Sync hidden fields
    syncHiddenFields();
}

function selectModel(modelType) {
    currentModelType = modelType;

    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.model === modelType);
    });

    // Show/hide NN-specific hyperparams
    const nnParams = document.querySelectorAll('.nn-only');
    nnParams.forEach(el => {
        el.style.display = modelType === 'neural_network' ? '' : 'none';
    });

    syncHiddenFields();
}

function syncHiddenFields() {
    const taskEl = document.getElementById('hidden-task-type');
    const modelEl = document.getElementById('hidden-model-type');
    const combinedEl = document.getElementById('hidden-training-task');

    if (taskEl) taskEl.value = currentTaskType;
    if (modelEl) modelEl.value = currentModelType;
    if (combinedEl) combinedEl.value = `${currentTaskType}_${currentModelType}`;
}

// ============================================================================
// Gather Full Training Config
// ============================================================================

function getTrainingConfig() {
    const hiddenLayersStr = document.getElementById('hp-hidden-layers')?.value || '512,256,128,64';
    const hiddenLayers = hiddenLayersStr.split(',').map(x => parseInt(x.trim()));

    const selectedFeatures = getSelectedFeatures();
    const totalAvailable = featureSelectionState.allFeatures.numeric.length +
                           featureSelectionState.allFeatures.categorical.length +
                           featureSelectionState.allFeatures.boolean.length;
    const useCustomSelection = selectedFeatures.length < totalAvailable && selectedFeatures.length > 0;

    const lowerPercentile = parseInt(document.getElementById('hp-lower-percentile')?.value || '90', 10);
    const upperPercentile = parseInt(document.getElementById('hp-upper-percentile')?.value || '100', 10);
    const nClusters = parseInt(document.getElementById('hp-n-clusters')?.value || '5', 10);
    const clusterThreshold = parseFloat(document.getElementById('hp-cluster-threshold')?.value || '0.5');

    return {
        task_type: currentTaskType,
        model_type: currentModelType,
        target: currentTaskType === 'lookalike' ? 'avg_monthly_revenue' : 'avg_monthly_revenue',
        device: 'mps',
        apple_chip: 'auto',
        model_preset: 'model_a',
        epochs: parseInt(document.getElementById('hp-epochs')?.value || '50'),
        batch_size: parseInt(document.getElementById('hp-batch-size')?.value || '4096'),
        learning_rate: parseFloat(document.getElementById('hp-learning-rate')?.value || '0.0001'),
        dropout: parseFloat(document.getElementById('hp-dropout')?.value || '0.2'),
        hidden_layers: hiddenLayers,
        embedding_dim: parseInt(document.getElementById('hp-embedding-dim')?.value || '16'),
        early_stopping_patience: parseInt(document.getElementById('hp-early-stopping')?.value || '10'),
        selected_features: useCustomSelection ? selectedFeatures : null,
        feature_selection_method: 'none',
        stg_lambda: 0.1,
        stg_sigma: 0.5,
        run_shap_validation: false,
        track_gradients: false,
        lookalike_lower_percentile: lowerPercentile,
        lookalike_upper_percentile: upperPercentile,
        n_clusters: nClusters,
        cluster_probability_threshold: clusterThreshold,
        network_filter: document.getElementById('hp-network-filter')?.value || null,
    };
}

// ============================================================================
// Validation
// ============================================================================

function validatePercentileBounds() {
    if (currentTaskType !== 'lookalike') return { valid: true };

    const lower = parseInt(document.getElementById('hp-lower-percentile')?.value || '90', 10);
    const upper = parseInt(document.getElementById('hp-upper-percentile')?.value || '100', 10);

    if (isNaN(lower) || lower < 1 || lower > 99) {
        return { valid: false, error: 'Lower percentile must be between 1 and 99' };
    }
    if (isNaN(upper) || upper < 1 || upper > 100) {
        return { valid: false, error: 'Upper percentile must be between 1 and 100' };
    }
    if (upper <= lower) {
        return { valid: false, error: 'Upper percentile must be greater than lower' };
    }
    return { valid: true };
}

function validateConfig() {
    const featureCount = getSelectedFeatures().length;
    if (featureCount < 3) {
        return { valid: false, error: 'At least 3 features must be selected for training.' };
    }

    const percentileCheck = validatePercentileBounds();
    if (!percentileCheck.valid) return percentileCheck;

    return { valid: true };
}

// ============================================================================
// System Info Loader
// ============================================================================

async function loadSystemInfo() {
    try {
        const response = await fetch('/api/training/system-info');
        const info = await response.json();

        const chipEl = document.getElementById('sys-chip');
        const gpuEl = document.getElementById('sys-gpu-cores');
        const memEl = document.getElementById('sys-memory');
        const mpsEl = document.getElementById('sys-mps');

        if (info.detected_chip && info.detected_chip !== 'unknown') {
            const specs = APPLE_CHIP_SPECS[info.detected_chip];
            if (chipEl) chipEl.textContent = specs?.name || info.chip_name || 'Apple Silicon';
            if (gpuEl) gpuEl.textContent = specs?.gpuCores || info.gpu_cores || '-';
            if (memEl) memEl.textContent = info.total_memory || specs?.memory || '-';
        } else {
            if (chipEl) chipEl.textContent = info.chip_name || 'Apple Silicon';
            if (gpuEl) gpuEl.textContent = info.gpu_cores || '-';
            if (memEl) memEl.textContent = info.total_memory || '-';
        }

        if (mpsEl) {
            mpsEl.textContent = info.mps_available ? 'Available' : 'Not Available';
            mpsEl.classList.add(info.mps_available ? 'available' : '');
        }
    } catch (error) {
        console.error('Failed to load system info:', error);
    }
}

// ============================================================================
// Percentile Hint Updater
// ============================================================================

function updatePercentileHint() {
    const lower = parseInt(document.getElementById('hp-lower-percentile')?.value || '90', 10);
    const upper = parseInt(document.getElementById('hp-upper-percentile')?.value || '100', 10);
    const hintEl = document.getElementById('percentile-hint-text');
    const errorEl = document.getElementById('percentile-error-text');

    if (!hintEl) return;

    if (upper <= lower) {
        if (errorEl) {
            errorEl.textContent = 'Upper must be greater than lower';
            errorEl.style.display = 'block';
        }
    } else {
        if (errorEl) errorEl.style.display = 'none';
        const range = upper - lower;
        hintEl.textContent = `Top ${range}% of sites by revenue (p${lower}-p${upper}) labeled as top performers`;
    }
}

// ============================================================================
// Page Initialization
// ============================================================================

function initTrainingConfig() {
    // Load features
    loadTrainingFeatures();

    // Load system info
    loadSystemInfo();

    // Set initial task/model selection
    selectTask('regression');
    selectModel('xgboost');

    // Wire feature search
    const searchInput = document.getElementById('feature-search-input');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => filterFeatures(e.target.value));
    }

    // Wire feature tabs
    document.querySelectorAll('.feature-tab').forEach(tab => {
        tab.addEventListener('click', () => switchFeatureTab(tab.dataset.type));
    });

    // Wire hyperparameter toggle
    const hpToggle = document.getElementById('hyperparam-toggle');
    if (hpToggle) {
        hpToggle.addEventListener('click', () => {
            hpToggle.classList.toggle('expanded');
            document.getElementById('hyperparam-body')?.classList.toggle('expanded');
        });
    }

    // Wire percentile inputs
    const lowerInput = document.getElementById('hp-lower-percentile');
    const upperInput = document.getElementById('hp-upper-percentile');
    if (lowerInput) {
        lowerInput.addEventListener('input', updatePercentileHint);
        lowerInput.addEventListener('change', updatePercentileHint);
    }
    if (upperInput) {
        upperInput.addEventListener('input', updatePercentileHint);
        upperInput.addEventListener('change', updatePercentileHint);
    }

    // Wire cluster threshold slider
    const threshSlider = document.getElementById('hp-cluster-threshold');
    const threshValue = document.getElementById('cluster-threshold-display');
    if (threshSlider && threshValue) {
        threshSlider.addEventListener('input', function() {
            const val = parseFloat(this.value);
            threshValue.textContent = `${val.toFixed(1)} (${Math.round(val * 100)}% confidence)`;
        });
    }
}
