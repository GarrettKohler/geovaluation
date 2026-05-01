/**
 * Experiment Catalog — Loads, renders, and manages the experiment gallery.
 * Used on the home page right column.
 *
 * Depends on: shared-utils.js (getFeatureDisplayName)
 */

// ============================================================================
// State
// ============================================================================

let experimentCatalog = [];
let catalogFilterTask = '';
let catalogSortBy = 'newest';

// ============================================================================
// Load Catalog from API
// ============================================================================

async function loadExperimentCatalog() {
    try {
        const response = await fetch('/api/experiments/catalog');
        const data = await response.json();
        experimentCatalog = data.experiments || [];
        renderExperimentGallery();
    } catch (err) {
        console.error('Failed to load experiment catalog:', err);
        const gallery = document.getElementById('experiment-gallery');
        if (gallery) {
            gallery.innerHTML = '<div class="experiment-empty">Failed to load experiments.</div>';
        }
    }
}

// ============================================================================
// Filter & Sort
// ============================================================================

function getFilteredCatalog() {
    let filtered = [...experimentCatalog];

    // Task type filter
    if (catalogFilterTask) {
        filtered = filtered.filter(exp => exp.task_type === catalogFilterTask);
    }

    // Sort
    if (catalogSortBy === 'newest') {
        filtered.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
    } else if (catalogSortBy === 'oldest') {
        filtered.sort((a, b) => (a.created_at || '').localeCompare(b.created_at || ''));
    } else if (catalogSortBy === 'best') {
        filtered.sort((a, b) => {
            const metricA = getPrimaryMetricValue(a);
            const metricB = getPrimaryMetricValue(b);
            return metricB - metricA;
        });
    }

    return filtered;
}

function getPrimaryMetricValue(exp) {
    const m = exp.test_metrics || {};
    if (exp.task_type === 'lookalike') return m.test_r2 || 0;
    if (exp.task_type === 'clustering') return m.silhouette_score || m.test_r2 || 0;
    return m.test_r2 || 0;
}

// ============================================================================
// Render Gallery
// ============================================================================

function renderExperimentGallery() {
    const gallery = document.getElementById('experiment-gallery');
    if (!gallery) return;

    const filtered = getFilteredCatalog();

    // Update count
    const countEl = document.getElementById('experiment-count');
    if (countEl) countEl.textContent = filtered.length;

    if (filtered.length === 0) {
        gallery.innerHTML = `
            <div class="experiment-empty">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                    <path d="M2 17l10 5 10-5"/>
                    <path d="M2 12l10 5 10-5"/>
                </svg>
                <div>${catalogFilterTask ? 'No matching experiments.' : 'No experiments yet. Train a model to get started.'}</div>
            </div>
        `;
        return;
    }

    gallery.innerHTML = filtered.map(exp => renderExperimentGalleryCard(exp)).join('');
}

function renderExperimentGalleryCard(exp) {
    const incompleteClass = exp.is_complete ? '' : ' incomplete';

    // Model badge
    const modelBadge = exp.model_type === 'neural_network'
        ? '<span class="exp-badge exp-badge-nn">NN</span>'
        : exp.model_type === 'xgboost'
            ? '<span class="exp-badge exp-badge-xgb">XGB</span>'
            : '<span class="exp-badge exp-badge-cluster">DEC</span>';

    // Task badge
    const taskBadge = exp.task_type === 'regression'
        ? '<span class="exp-badge exp-badge-regression">Regression</span>'
        : exp.task_type === 'lookalike'
            ? '<span class="exp-badge exp-badge-lookalike">Lookalike</span>'
            : '<span class="exp-badge exp-badge-clustering">Clustering</span>';

    // Date
    const dateStr = exp.created_at
        ? new Date(exp.created_at).toLocaleDateString('en-US', {
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
          })
        : 'Unknown date';

    // Metrics
    const metricsHtml = formatExperimentMetrics(exp);

    // Feature count
    const fc = exp.feature_count || {};
    const totalFeatures = (fc.numeric || 0) + (fc.categorical || 0) + (fc.boolean || 0);

    // Artifact badges
    let artifactBadges = '';
    if (exp.has_shap) artifactBadges += '<span class="exp-badge exp-badge-artifact">SHAP</span>';
    if (exp.has_predictions) artifactBadges += '<span class="exp-badge exp-badge-artifact">Predictions</span>';
    if (!exp.is_complete) artifactBadges += '<span class="exp-badge exp-badge-artifact" style="color: var(--kepler-warning);">(incomplete)</span>';

    const networkInfo = exp.network_filter ? `<span>${exp.network_filter}</span>` : '';

    return `
        <div class="experiment-gallery-card${incompleteClass}" onclick="navigateToExperiment('${exp.job_id}')">
            <div class="exp-card-top">
                <div class="exp-card-badges">${modelBadge}${taskBadge}</div>
                <span class="exp-card-date">${dateStr}</span>
            </div>
            <div class="exp-card-metrics">${metricsHtml}</div>
            <div class="exp-card-footer">
                <span>${totalFeatures} features</span>
                ${networkInfo}
                ${artifactBadges}
            </div>
        </div>
    `;
}

function formatExperimentMetrics(exp) {
    const m = exp.test_metrics || {};
    if (!m || Object.keys(m).length === 0) {
        return '<div class="exp-card-metric"><span class="metric-label">No metrics</span></div>';
    }

    if (exp.task_type === 'lookalike') {
        const auc = m.test_r2 != null ? m.test_r2.toFixed(4) : '\u2014';
        const f1 = m.test_f1 != null ? m.test_f1.toFixed(4) : '\u2014';
        return `
            <div class="exp-card-metric">
                <span class="metric-label">ROC-AUC</span>
                <span class="metric-value">${auc}</span>
            </div>
            <div class="exp-card-metric">
                <span class="metric-label">F1</span>
                <span class="metric-value">${f1}</span>
            </div>
        `;
    } else if (exp.task_type === 'clustering') {
        const sil = m.silhouette_score != null ? m.silhouette_score.toFixed(3) : (m.test_r2 != null ? m.test_r2.toFixed(3) : '\u2014');
        return `
            <div class="exp-card-metric">
                <span class="metric-label">Silhouette</span>
                <span class="metric-value">${sil}</span>
            </div>
        `;
    } else {
        const r2 = m.test_r2 != null ? m.test_r2.toFixed(4) : '\u2014';
        const mae = m.test_mae != null ? '$' + Math.round(m.test_mae).toLocaleString() : '\u2014';
        return `
            <div class="exp-card-metric">
                <span class="metric-label">R\u00B2</span>
                <span class="metric-value">${r2}</span>
            </div>
            <div class="exp-card-metric">
                <span class="metric-label">MAE</span>
                <span class="metric-value">${mae}</span>
            </div>
        `;
    }
}

// ============================================================================
// Navigation
// ============================================================================

function navigateToExperiment(jobId) {
    window.location.href = `/map/${jobId}`;
}

// ============================================================================
// Filter/Sort Event Handlers
// ============================================================================

function initExperimentCatalog() {
    // Load catalog
    loadExperimentCatalog();

    // Wire filter dropdown
    const filterSelect = document.getElementById('catalog-filter-task');
    if (filterSelect) {
        filterSelect.addEventListener('change', (e) => {
            catalogFilterTask = e.target.value;
            renderExperimentGallery();
        });
    }

    // Wire sort dropdown
    const sortSelect = document.getElementById('catalog-sort');
    if (sortSelect) {
        sortSelect.addEventListener('change', (e) => {
            catalogSortBy = e.target.value;
            renderExperimentGallery();
        });
    }
}
