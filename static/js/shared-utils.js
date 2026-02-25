/**
 * Shared utilities used by both the home page and the map page.
 * Extracted from index.html to avoid duplication.
 */

// ============================================================================
// Feature Display Names
// ============================================================================

const featureDisplayNames = {
    // Numeric
    'rs_NVIs': 'NVI Momentum',
    'rs_Revenue': 'Revenue Momentum',
    'avg_monthly_revenue': 'Avg Monthly Revenue',
    'log_total_revenue': 'Log Total Revenue',
    'log_nearest_site_distance_mi': 'Nearest Site Distance',
    'log_min_distance_to_interstate_mi': 'Log Min Distance to Interstate (miles)',
    'log_min_distance_to_kroger_mi': 'Log Distance to Kroger (miles)',
    'log_min_distance_to_mcdonalds_mi': 'Log Distance to McDonalds (miles)',
    'log_min_distance_to_walmart_mi': 'Log Distance to Walmart (miles)',
    'log_min_distance_to_target_mi': 'Log Distance to Target (miles)',
    'log_avg_household_income': 'Log Avg Household Income',
    'median_age': 'Median Age',
    'pct_female': '% Female',
    'avg_monthly_monthly_impressions': 'Avg Monthly Impressions',
    'avg_monthly_monthly_nvis': 'Avg Monthly NVIs',
    'avg_monthly_monthly_impressions_per_screen': 'Avg Impr/Screen',
    'avg_monthly_monthly_nvis_per_screen': 'Avg NVIs/Screen',
    'avg_monthly_monthly_revenue_per_screen': 'Avg Rev/Screen',
    'log_total_monthly_impressions': 'Log Total Impressions',
    'log_total_monthly_nvis': 'Log Total NVIs',
    'log_total_monthly_impressions_per_screen': 'Log Impr/Screen',
    'log_total_monthly_nvis_per_screen': 'Log NVIs/Screen',
    'log_total_monthly_revenue_per_screen': 'Log Rev/Screen',
    'pct_african_american': '% African American',
    'pct_asian': '% Asian',
    'pct_hispanic': '% Hispanic',
    'screen_count': 'Screen Count',
    'dma_rank': 'DMA Rank',
    'active_months': 'Active Months',
    // Categorical
    'network': 'Network',
    'program': 'Program',
    'experience_type': 'Experience Type',
    'hardware_type': 'Hardware Type',
    'retailer': 'Retailer',
    'brand_fuel': 'Fuel Brand',
    'brand_c_store': 'C-Store Brand',
};

function getFeatureDisplayName(raw) {
    if (featureDisplayNames[raw]) return featureDisplayNames[raw];
    // Boolean features: strip prefix and _encoded suffix
    let name = raw.replace(/_encoded$/, '');
    if (name.startsWith('r_')) name = name.slice(2);
    if (name.startsWith('c_')) name = name.slice(2);
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ============================================================================
// Value Formatting
// ============================================================================

function formatValue(label, value) {
    if (value === null || value === undefined || value === '') {
        return { display: '\u2014', cssClass: 'empty' };
    }

    // Currency formatting
    const currencyFields = ['Avg Monthly Revenue', 'Total Revenue', 'Avg Household Income'];
    if (currencyFields.includes(label)) {
        const num = parseFloat(value);
        if (!isNaN(num)) {
            return {
                display: num.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }),
                cssClass: 'currency'
            };
        }
    }

    // Percentage formatting
    if (label.startsWith('%') || label.includes('Score')) {
        const num = parseFloat(value);
        if (!isNaN(num)) {
            if (label === 'Revenue Score') {
                return { display: `${(num * 100).toFixed(1)}%`, cssClass: '' };
            }
            return { display: `${num.toFixed(1)}%`, cssClass: '' };
        }
    }

    // Number formatting
    const numberFields = ['Avg Daily Impressions', 'Avg Daily Visits', 'Screen Count', 'Active Months', 'DMA Rank'];
    if (numberFields.includes(label)) {
        const num = parseFloat(value);
        if (!isNaN(num)) {
            return { display: num.toLocaleString('en-US', { maximumFractionDigits: 0 }), cssClass: '' };
        }
    }

    // Latency formatting
    if (label === 'Avg Latency') {
        const num = parseFloat(value);
        if (!isNaN(num)) return { display: `${num.toFixed(1)} ms`, cssClass: '' };
    }

    // Coordinate formatting
    if (label === 'Latitude' || label === 'Longitude') {
        const num = parseFloat(value);
        if (!isNaN(num)) return { display: num.toFixed(6), cssClass: '' };
    }

    // Age formatting
    if (label === 'Median Age') {
        const num = parseFloat(value);
        if (!isNaN(num)) return { display: num.toFixed(1), cssClass: '' };
    }

    // Boolean formatting
    if (typeof value === 'boolean') {
        return { display: value ? 'Yes' : 'No', cssClass: value ? 'boolean-true' : 'boolean-false' };
    }
    const strVal = String(value).toLowerCase();
    if (strVal === 'true' || strVal === 'yes') return { display: 'Yes', cssClass: 'boolean-true' };
    if (strVal === 'false' || strVal === 'no') return { display: 'No', cssClass: 'boolean-false' };
    if (strVal === 'unknown') return { display: 'Unknown', cssClass: 'empty' };

    return { display: String(value), cssClass: '' };
}

// ============================================================================
// Apple Silicon Chip Specifications
// ============================================================================

const APPLE_CHIP_SPECS = {
    'm1':       { name: 'Apple M1', gpuCores: 8, memory: '8-16 GB', maxBatch: 4096, tier: 1 },
    'm1_pro':   { name: 'Apple M1 Pro', gpuCores: 16, memory: '16-32 GB', maxBatch: 8192, tier: 2 },
    'm1_max':   { name: 'Apple M1 Max', gpuCores: 32, memory: '32-64 GB', maxBatch: 16384, tier: 3 },
    'm1_ultra': { name: 'Apple M1 Ultra', gpuCores: 64, memory: '64-128 GB', maxBatch: 32768, tier: 4 },
    'm2':       { name: 'Apple M2', gpuCores: 10, memory: '8-24 GB', maxBatch: 4096, tier: 1 },
    'm2_pro':   { name: 'Apple M2 Pro', gpuCores: 19, memory: '16-32 GB', maxBatch: 8192, tier: 2 },
    'm2_max':   { name: 'Apple M2 Max', gpuCores: 38, memory: '32-96 GB', maxBatch: 16384, tier: 3 },
    'm2_ultra': { name: 'Apple M2 Ultra', gpuCores: 76, memory: '64-192 GB', maxBatch: 32768, tier: 4 },
    'm3':       { name: 'Apple M3', gpuCores: 10, memory: '8-24 GB', maxBatch: 4096, tier: 1 },
    'm3_pro':   { name: 'Apple M3 Pro', gpuCores: 18, memory: '18-36 GB', maxBatch: 8192, tier: 2 },
    'm3_max':   { name: 'Apple M3 Max', gpuCores: 40, memory: '36-128 GB', maxBatch: 16384, tier: 3 },
    'm4':       { name: 'Apple M4', gpuCores: 10, memory: '16-32 GB', maxBatch: 8192, tier: 2 },
    'm4_pro':   { name: 'Apple M4 Pro', gpuCores: 20, memory: '24-64 GB', maxBatch: 16384, tier: 3 },
    'm4_max':   { name: 'Apple M4 Max', gpuCores: 40, memory: '36-128 GB', maxBatch: 32768, tier: 4 },
};
