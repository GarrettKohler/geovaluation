/**
 * Training Stream — SSE connection, progress UI, Chart.js, training modal.
 * Used on the home page during active training.
 *
 * Depends on: shared-utils.js, training-config.js (for getTrainingConfig, validateConfig)
 */

// ============================================================================
// State
// ============================================================================

let trainingEventSource = null;
let currentJobId = null;
let trainingStartTime = null;

// Chart instances
let lossChart = null;
let performanceChart = null;
let weightHistogramChart = null;
let biasHistogramChart = null;
let trainingChartData = { epochs: [], trainLoss: [], valLoss: [], performance: [] };

// ============================================================================
// Start Training
// ============================================================================

async function startTraining() {
    const validation = validateConfig();
    if (!validation.valid) {
        alert(validation.error);
        return;
    }

    const config = getTrainingConfig();
    const trainBtn = document.getElementById('btn-start-training');

    // Disable button
    trainBtn.disabled = true;
    trainBtn.innerHTML = '<div class="spinner" style="width:18px;height:18px;margin:0;"></div> Starting...';

    try {
        const response = await fetch('/api/training/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const result = await response.json();

        if (result.success) {
            currentJobId = result.job_id;
            trainingStartTime = Date.now();
            showTrainingModal();
            initializeTrainingCharts();
            connectToProgressStream();
        } else {
            alert('Failed to start training: ' + result.error);
            resetTrainButton();
        }
    } catch (error) {
        console.error('Training start error:', error);
        alert('Failed to start training: ' + error.message);
        resetTrainButton();
    }
}

function resetTrainButton() {
    const trainBtn = document.getElementById('btn-start-training');
    if (trainBtn) {
        trainBtn.disabled = false;
        trainBtn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="5 3 19 12 5 21 5 3"/>
        </svg> Start Training`;
    }
}

// ============================================================================
// Training Modal — Show/Hide
// ============================================================================

function showTrainingModal() {
    const overlay = document.getElementById('training-modal-overlay');
    if (overlay) overlay.classList.add('active');

    // Set modal header info
    const titleEl = document.getElementById('modal-training-title');
    const subtitleEl = document.getElementById('modal-training-subtitle');

    const taskLabels = {
        regression: 'Revenue Prediction',
        lookalike: 'Lookalike Classification',
        clustering: 'Cluster Analysis'
    };
    const modelLabels = {
        xgboost: 'XGBoost',
        neural_network: 'Neural Network'
    };

    if (titleEl) titleEl.textContent = 'Training in Progress';
    if (subtitleEl) subtitleEl.textContent = `${taskLabels[currentTaskType] || currentTaskType} \u2014 ${modelLabels[currentModelType] || currentModelType}`;

    // Show progress phase, hide complete phase
    const progressPhase = document.getElementById('modal-phase-progress');
    const completePhase = document.getElementById('modal-phase-complete');
    if (progressPhase) progressPhase.style.display = 'block';
    if (completePhase) completePhase.style.display = 'none';

    // Configure metric visibility based on task
    const isLookalike = currentTaskType === 'lookalike';
    const isClustering = currentTaskType === 'clustering';

    // Hide/show appropriate metric cards
    const regressionMetrics = document.querySelectorAll('.metric-regression');
    const classificationMetrics = document.querySelectorAll('.metric-classification');
    regressionMetrics.forEach(el => el.style.display = isLookalike || isClustering ? 'none' : '');
    classificationMetrics.forEach(el => el.style.display = isLookalike ? '' : 'none');

    // Clear log
    const logEl = document.getElementById('modal-progress-log');
    if (logEl) logEl.innerHTML = '';
    addLogEntry('Training job started...', 'normal');
}

function hideTrainingModal() {
    const overlay = document.getElementById('training-modal-overlay');
    if (overlay) overlay.classList.remove('active');
}

// ============================================================================
// SSE Connection
// ============================================================================

function connectToProgressStream() {
    if (trainingEventSource) {
        trainingEventSource.close();
    }

    trainingEventSource = new EventSource('/api/training/stream');

    trainingEventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            if (currentJobId && data.job_id && data.job_id !== currentJobId) return;
            updateProgressUI(data);

            if (data.status === 'completed' || data.status === 'error' || data.status === 'stopped' || data.status === 'stream_end') {
                trainingEventSource.close();
                trainingEventSource = null;

                if (data.status === 'completed') {
                    showTrainingComplete(data);
                    // Refresh experiment catalog
                    if (typeof loadExperimentCatalog === 'function') loadExperimentCatalog();
                } else if (data.status === 'error') {
                    addLogEntry('Error: ' + data.message, 'error');
                } else if (data.status === 'stopped') {
                    addLogEntry('Training stopped by user', 'normal');
                }
                resetTrainButton();
            }
        } catch (err) {
            console.error('SSE message processing error:', err);
        }
    };

    trainingEventSource.onerror = function(error) {
        console.error('SSE error:', error);
        addLogEntry('Connection error - checking status...', 'error');
        trainingEventSource.close();
        trainingEventSource = null;
    };
}

// ============================================================================
// Progress UI Updates
// ============================================================================

function updateProgressUI(data) {
    if (data.status === 'stream_end') return;

    // Status badge
    const statusEl = document.getElementById('modal-status');
    if (statusEl) {
        statusEl.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
        statusEl.className = 'modal-progress-status ' + data.status;
    }

    // Epoch counter
    if (data.epoch !== undefined && data.total_epochs !== undefined) {
        const epochEl = document.getElementById('modal-epoch');
        if (epochEl) epochEl.textContent = `${data.epoch}/${data.total_epochs}`;
    }

    // Progress bar
    const progress = (data.total_epochs && data.total_epochs > 0) ? (data.epoch / data.total_epochs) * 100 : 0;
    const barEl = document.getElementById('modal-progress-bar');
    if (barEl) barEl.style.width = progress + '%';

    // Metrics
    if (data.train_loss > 0) {
        setMetricText('modal-metric-train-loss', data.train_loss.toFixed(4));
    }
    if (data.val_loss > 0) {
        setMetricText('modal-metric-val-loss', data.val_loss.toFixed(4));
    }

    const isLookalike = currentTaskType === 'lookalike';

    if (!isLookalike) {
        if (data.val_mae > 0) setMetricText('modal-metric-mae', '$' + data.val_mae.toLocaleString(undefined, {maximumFractionDigits: 0}));
        if (data.val_smape > 0) setMetricText('modal-metric-smape', data.val_smape.toFixed(2) + '%');
        if (data.val_rmse > 0) setMetricText('modal-metric-rmse', '$' + data.val_rmse.toLocaleString(undefined, {maximumFractionDigits: 0}));
    }

    if (data.val_r2 !== undefined && data.val_r2 !== 0) {
        setMetricText('modal-metric-r2', data.val_r2.toFixed(4));
    }
    if (data.val_f1 !== undefined && data.val_f1 > 0) {
        setMetricText('modal-metric-f1', data.val_f1.toFixed(4));
    }
    if (data.val_logloss !== undefined && data.val_logloss > 0) {
        setMetricText('modal-metric-logloss', data.val_logloss.toFixed(4));
    }

    // Log entry
    if (data.message) {
        const logClass = data.status === 'completed' ? 'success' : (data.status === 'error' ? 'error' : 'normal');
        addLogEntry(data.message, logClass);
    }

    // Charts
    if (data.epoch > 0 && data.train_loss > 0) {
        updateTrainingCharts(data);
    }
}

function setMetricText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

// ============================================================================
// Chart.js — Training Charts
// ============================================================================

function initializeTrainingCharts() {
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js not available');
        return;
    }

    trainingChartData = { epochs: [], trainLoss: [], valLoss: [], performance: [] };

    const isLookalike = currentTaskType === 'lookalike';

    // Update chart titles
    setMetricText('modal-loss-chart-title', (currentModelType === 'neural_network' && !isLookalike) ? 'Huber Loss' : 'Loss Curve');
    setMetricText('modal-perf-chart-title', isLookalike ? 'ROC-AUC' : 'R\u00B2 Score');

    const chartDefaults = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        plugins: {
            legend: {
                display: true,
                position: 'top',
                labels: { color: '#4a4a48', font: { size: 10 }, boxWidth: 12, padding: 8 }
            }
        },
        scales: {
            x: { display: true, grid: { color: 'rgba(200, 197, 190, 0.5)' }, ticks: { color: '#8b8985', font: { size: 9 } } },
            y: { display: true, grid: { color: 'rgba(200, 197, 190, 0.5)' }, ticks: { color: '#8b8985', font: { size: 9 } } }
        }
    };

    // Destroy existing
    if (lossChart) lossChart.destroy();
    if (performanceChart) performanceChart.destroy();
    if (weightHistogramChart) { weightHistogramChart.destroy(); weightHistogramChart = null; }
    if (biasHistogramChart) { biasHistogramChart.destroy(); biasHistogramChart = null; }

    // Loss chart
    const lossCtx = document.getElementById('modal-loss-chart')?.getContext('2d');
    if (lossCtx) {
        lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Train Loss', data: [], borderColor: '#1a7457', backgroundColor: 'rgba(26, 116, 87, 0.08)', borderWidth: 2, pointRadius: 0, tension: 0.3 },
                    { label: 'Val Loss', data: [], borderColor: '#d9363e', backgroundColor: 'rgba(217, 54, 62, 0.08)', borderWidth: 2, pointRadius: 0, tension: 0.3 }
                ]
            },
            options: chartDefaults
        });
    }

    // Performance chart
    const perfCtx = document.getElementById('modal-perf-chart')?.getContext('2d');
    if (perfCtx) {
        performanceChart = new Chart(perfCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: isLookalike ? 'ROC-AUC' : 'R\u00B2',
                    data: [],
                    borderColor: '#1a7457',
                    backgroundColor: 'rgba(26, 116, 87, 0.08)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                    fill: true
                }]
            },
            options: {
                ...chartDefaults,
                scales: {
                    ...chartDefaults.scales,
                    y: { ...chartDefaults.scales.y, min: 0, max: 1, ticks: { color: '#8b8985', font: { size: 9 }, callback: v => v.toFixed(1) } }
                }
            }
        });
    }

    // Histograms (NN only)
    const histRow = document.getElementById('modal-histogram-row');
    if (histRow) {
        if (currentModelType === 'neural_network') {
            histRow.style.display = 'grid';
            const histDefaults = {
                responsive: true, maintainAspectRatio: false, animation: { duration: 0 },
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: true, grid: { color: 'rgba(200, 197, 190, 0.5)' }, ticks: { color: '#8b8985', font: { size: 8 }, maxTicksLimit: 5 } },
                    y: { display: true, grid: { color: 'rgba(200, 197, 190, 0.5)' }, ticks: { color: '#8b8985', font: { size: 8 } } }
                }
            };

            const wCtx = document.getElementById('modal-weight-hist')?.getContext('2d');
            if (wCtx) {
                weightHistogramChart = new Chart(wCtx, {
                    type: 'bar',
                    data: { labels: [], datasets: [{ data: [], backgroundColor: 'rgba(37, 99, 235, 0.4)', borderColor: 'rgba(37, 99, 235, 0.8)', borderWidth: 1 }] },
                    options: histDefaults
                });
            }

            const bCtx = document.getElementById('modal-bias-hist')?.getContext('2d');
            if (bCtx) {
                biasHistogramChart = new Chart(bCtx, {
                    type: 'bar',
                    data: { labels: [], datasets: [{ data: [], backgroundColor: 'rgba(196, 138, 26, 0.4)', borderColor: 'rgba(196, 138, 26, 0.8)', borderWidth: 1 }] },
                    options: histDefaults
                });
            }
        } else {
            histRow.style.display = 'none';
        }
    }
}

function updateTrainingCharts(data) {
    if (typeof Chart === 'undefined' || !lossChart || !performanceChart) return;

    trainingChartData.epochs.push(data.epoch);
    trainingChartData.trainLoss.push(data.train_loss);
    trainingChartData.valLoss.push(data.val_loss);
    trainingChartData.performance.push(data.val_r2 || 0);

    lossChart.data.labels = trainingChartData.epochs;
    lossChart.data.datasets[0].data = trainingChartData.trainLoss;
    lossChart.data.datasets[1].data = trainingChartData.valLoss;
    lossChart.update('none');

    performanceChart.data.labels = trainingChartData.epochs;
    performanceChart.data.datasets[0].data = trainingChartData.performance;
    performanceChart.update('none');

    // Histograms
    if (data.weight_histograms && weightHistogramChart && biasHistogramChart) {
        const wh = data.weight_histograms;
        if (wh.weights) {
            const labels = wh.weights.counts.map((_, i) =>
                ((wh.weights.bin_edges[i] + wh.weights.bin_edges[i+1]) / 2).toFixed(3));
            weightHistogramChart.data.labels = labels;
            weightHistogramChart.data.datasets[0].data = wh.weights.counts;
            weightHistogramChart.update('none');
        }
        if (wh.biases) {
            const labels = wh.biases.counts.map((_, i) =>
                ((wh.biases.bin_edges[i] + wh.biases.bin_edges[i+1]) / 2).toFixed(3));
            biasHistogramChart.data.labels = labels;
            biasHistogramChart.data.datasets[0].data = wh.biases.counts;
            biasHistogramChart.update('none');
        }
    }
}

// ============================================================================
// Log
// ============================================================================

function addLogEntry(message, type) {
    const log = document.getElementById('modal-progress-log');
    if (!log) return;
    const entry = document.createElement('div');
    entry.className = 'log-entry ' + type;
    const timestamp = new Date().toLocaleTimeString();
    entry.textContent = `[${timestamp}] ${message}`;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
}

// ============================================================================
// Stop Training
// ============================================================================

async function stopTraining() {
    try {
        const response = await fetch('/api/training/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_id: currentJobId })
        });
        const result = await response.json();
        addLogEntry(result.message, 'normal');
    } catch (error) {
        console.error('Stop training error:', error);
        addLogEntry('Failed to stop training: ' + error.message, 'error');
    }
}

// ============================================================================
// Training Complete — Switch to results phase
// ============================================================================

function showTrainingComplete(data) {
    const titleEl = document.getElementById('modal-training-title');
    if (titleEl) titleEl.textContent = 'Training Complete';

    // Switch phases
    const progressPhase = document.getElementById('modal-phase-progress');
    const completePhase = document.getElementById('modal-phase-complete');
    if (progressPhase) progressPhase.style.display = 'none';
    if (completePhase) completePhase.style.display = 'block';

    // Extract metrics
    let metrics = data.final_metrics || {
        test_mae: data.val_mae || 0,
        test_smape: data.val_smape || 0,
        test_rmse: data.val_rmse || 0,
        test_r2: data.val_r2 || 0,
        test_loss: data.val_loss || 0,
        test_f1: data.val_f1 || 0,
        test_logloss: data.val_logloss || 0,
        elapsed_time: data.elapsed_time
    };

    const isLookalike = currentTaskType === 'lookalike';
    const isClustering = currentTaskType === 'clustering';

    // Populate complete metrics
    if (isLookalike) {
        setMetricText('complete-metric-primary-label', 'ROC-AUC');
        setMetricText('complete-metric-primary', metrics.test_r2 !== undefined ? metrics.test_r2.toFixed(4) : '\u2014');
        setMetricText('complete-metric-secondary-label', 'F1 Score');
        setMetricText('complete-metric-secondary', metrics.test_f1 !== undefined ? metrics.test_f1.toFixed(4) : '\u2014');
    } else if (isClustering) {
        setMetricText('complete-metric-primary-label', 'Silhouette');
        setMetricText('complete-metric-primary', metrics.test_r2 !== undefined ? metrics.test_r2.toFixed(3) : '\u2014');
        setMetricText('complete-metric-secondary-label', 'Clusters');
        setMetricText('complete-metric-secondary', metrics.n_clusters || '\u2014');
    } else {
        setMetricText('complete-metric-primary-label', 'Test R\u00B2');
        setMetricText('complete-metric-primary', metrics.test_r2 !== undefined ? metrics.test_r2.toFixed(4) : '\u2014');
        setMetricText('complete-metric-secondary-label', 'Test MAE');
        setMetricText('complete-metric-secondary', metrics.test_mae ? '$' + Math.round(metrics.test_mae).toLocaleString() : '\u2014');
    }

    if (metrics.test_loss) {
        setMetricText('complete-metric-loss', metrics.test_loss.toFixed(4));
    }
    if (metrics.elapsed_time || data.elapsed_time) {
        const elapsed = metrics.elapsed_time || data.elapsed_time;
        const minutes = Math.floor(elapsed / 60);
        const seconds = Math.round(elapsed % 60);
        setMetricText('complete-metric-time', minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`);
    }

    // Revenue Prediction metrics (Phase 2, lookalike tasks only)
    const revResults = document.getElementById('revenue-prediction-results');
    if (revResults) {
        if (metrics.has_revenue_predictions) {
            revResults.style.display = '';
            if (metrics.revenue_r2 !== undefined) {
                setMetricText('complete-revenue-r2', metrics.revenue_r2.toFixed(4));
            }
            if (metrics.revenue_mae) {
                setMetricText('complete-revenue-mae', '$' + Math.round(metrics.revenue_mae).toLocaleString());
            }
            if (metrics.revenue_rmse) {
                setMetricText('complete-revenue-rmse', '$' + Math.round(metrics.revenue_rmse).toLocaleString());
            }
            if (metrics.revenue_sites_scored) {
                setMetricText('complete-revenue-sites', metrics.revenue_sites_scored.toLocaleString());
            }
        } else {
            revResults.style.display = 'none';
        }
    }

    // Show and wire "View on Map" button
    const mapBtn = document.getElementById('btn-view-on-map');
    if (mapBtn && currentJobId) {
        mapBtn.style.display = '';
        mapBtn.onclick = () => {
            window.location.href = `/map/${currentJobId}`;
        };
    }
}
