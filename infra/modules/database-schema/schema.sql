-- DOOH Site Optimization Database Schema
-- Supports: Similarity modeling, Causal inference, Classification

-- Enable extensions (idempotent)
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS dooh;

-----------------------------------------------------------
-- CORE TABLES
-----------------------------------------------------------

-- Sites: The 60k gas station advertising locations
CREATE TABLE IF NOT EXISTS dooh.sites (
    site_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255),

    -- Location (PostGIS geometry)
    location GEOMETRY(Point, 4326) NOT NULL,
    address VARCHAR(500),
    city VARCHAR(100),
    state VARCHAR(50),
    zip_code VARCHAR(20),
    market_region VARCHAR(100),

    -- Site characteristics
    site_type VARCHAR(50),  -- gas_station, convenience_store, etc.
    traffic_volume INTEGER,
    distance_to_highway_km DECIMAL(10,3),
    poi_density INTEGER,  -- Points of interest within 1km
    competitor_count INTEGER,

    -- Status
    is_active BOOLEAN DEFAULT FALSE,
    activation_date DATE,
    deactivation_date DATE,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Spatial index for location queries
CREATE INDEX IF NOT EXISTS idx_sites_location ON dooh.sites USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_sites_market_region ON dooh.sites (market_region);
CREATE INDEX IF NOT EXISTS idx_sites_is_active ON dooh.sites (is_active);

-----------------------------------------------------------
-- HARDWARE & CONTENT
-----------------------------------------------------------

-- Hardware installed at each site
CREATE TABLE IF NOT EXISTS dooh.hardware (
    hardware_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    site_id UUID NOT NULL REFERENCES dooh.sites(site_id),

    -- Hardware specs
    display_technology VARCHAR(50) NOT NULL,  -- LCD, LED, OLED, interactive
    screen_size_inches DECIMAL(5,2),
    resolution VARCHAR(20),
    brightness_nits INTEGER,

    -- Installation
    installed_date DATE NOT NULL,
    removed_date DATE,
    is_current BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hardware_site ON dooh.hardware (site_id);
CREATE INDEX IF NOT EXISTS idx_hardware_current ON dooh.hardware (site_id, is_current) WHERE is_current = TRUE;

-----------------------------------------------------------
-- CAMPAIGNS & REVENUE
-----------------------------------------------------------

-- Advertising campaigns
CREATE TABLE IF NOT EXISTS dooh.campaigns (
    campaign_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    site_id UUID NOT NULL REFERENCES dooh.sites(site_id),

    -- Campaign details
    advertiser_name VARCHAR(255),
    content_type VARCHAR(50),  -- static, video, interactive
    content_category VARCHAR(100),  -- automotive, retail, food, etc.
    loop_length_seconds INTEGER,
    cpm_floor DECIMAL(10,2),

    -- Timing
    start_date DATE NOT NULL,
    end_date DATE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_campaigns_site ON dooh.campaigns (site_id);
CREATE INDEX IF NOT EXISTS idx_campaigns_dates ON dooh.campaigns (start_date, end_date);

-- Daily revenue per site
CREATE TABLE IF NOT EXISTS dooh.revenue (
    revenue_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    site_id UUID NOT NULL REFERENCES dooh.sites(site_id),

    -- Revenue data
    revenue_date DATE NOT NULL,
    gross_revenue DECIMAL(12,2) NOT NULL,
    impressions INTEGER,
    fill_rate DECIMAL(5,4),

    -- Aggregated metrics
    campaigns_active INTEGER,
    avg_cpm DECIMAL(10,2),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(site_id, revenue_date)
);

CREATE INDEX IF NOT EXISTS idx_revenue_site_date ON dooh.revenue (site_id, revenue_date);
CREATE INDEX IF NOT EXISTS idx_revenue_date ON dooh.revenue (revenue_date);

-----------------------------------------------------------
-- FEATURES (Pre-computed for ML)
-----------------------------------------------------------

-- Feature snapshots for training (point-in-time)
CREATE TABLE IF NOT EXISTS dooh.feature_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    site_id UUID NOT NULL REFERENCES dooh.sites(site_id),
    snapshot_date DATE NOT NULL,

    -- Site features (copied for point-in-time correctness)
    traffic_volume INTEGER,
    distance_to_highway_km DECIMAL(10,3),
    poi_density INTEGER,
    competitor_count INTEGER,
    market_region VARCHAR(100),

    -- Hardware features (current as of snapshot)
    display_technology VARCHAR(50),
    screen_size_inches DECIMAL(5,2),

    -- Content features (aggregated)
    primary_content_type VARCHAR(50),
    primary_content_category VARCHAR(100),
    avg_loop_length_seconds INTEGER,
    avg_cpm_floor DECIMAL(10,2),

    -- Revenue features (trailing windows)
    revenue_30d DECIMAL(12,2),
    revenue_90d DECIMAL(12,2),
    revenue_trend_30d DECIMAL(8,4),  -- % change

    -- Market proxy features (from nearby sites)
    nearby_avg_revenue_30d DECIMAL(12,2),
    nearby_high_revenue_pct DECIMAL(5,4),
    nearby_site_count INTEGER,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(site_id, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_features_site_date ON dooh.feature_snapshots (site_id, snapshot_date);

-----------------------------------------------------------
-- MODEL OUTPUTS
-----------------------------------------------------------

-- Model training runs
CREATE TABLE IF NOT EXISTS dooh.model_runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Run metadata
    model_type VARCHAR(50) NOT NULL,  -- similarity, causal, classifier
    model_version VARCHAR(50),
    mlflow_run_id VARCHAR(100),

    -- Training params
    training_start_date DATE,
    training_end_date DATE,
    validation_start_date DATE,
    validation_end_date DATE,

    -- Metrics
    metrics JSONB,  -- PR-AUC, MCC, etc.

    -- Artifacts
    model_artifact_path VARCHAR(500),

    -- Status
    status VARCHAR(20) DEFAULT 'running',  -- running, completed, failed
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_model_runs_type ON dooh.model_runs (model_type, started_at DESC);

-- Site predictions/scores
CREATE TABLE IF NOT EXISTS dooh.predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    site_id UUID NOT NULL REFERENCES dooh.sites(site_id),
    run_id UUID NOT NULL REFERENCES dooh.model_runs(run_id),

    -- Scores from each model
    lookalike_score DECIMAL(8,6),
    success_probability DECIMAL(8,6),
    expected_uplift DECIMAL(12,2),
    uplift_lower_bound DECIMAL(12,2),
    uplift_upper_bound DECIMAL(12,2),
    uplift_confident BOOLEAN,

    -- Combined priority score
    priority_score DECIMAL(8,6),
    priority_rank INTEGER,

    -- Recommendation
    recommended_action VARCHAR(500),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_site ON dooh.predictions (site_id);
CREATE INDEX IF NOT EXISTS idx_predictions_run ON dooh.predictions (run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_priority ON dooh.predictions (run_id, priority_rank);

-----------------------------------------------------------
-- VIEWS
-----------------------------------------------------------

-- Current site status with latest hardware
CREATE OR REPLACE VIEW dooh.v_site_current AS
SELECT
    s.*,
    h.display_technology,
    h.screen_size_inches,
    h.resolution,
    h.installed_date as hardware_installed_date
FROM dooh.sites s
LEFT JOIN dooh.hardware h ON s.site_id = h.site_id AND h.is_current = TRUE;

-- Latest predictions per site
CREATE OR REPLACE VIEW dooh.v_latest_predictions AS
SELECT DISTINCT ON (p.site_id)
    p.*,
    s.name as site_name,
    s.market_region,
    s.is_active
FROM dooh.predictions p
JOIN dooh.sites s ON p.site_id = s.site_id
JOIN dooh.model_runs mr ON p.run_id = mr.run_id
WHERE mr.status = 'completed'
ORDER BY p.site_id, p.created_at DESC;

-----------------------------------------------------------
-- FUNCTIONS
-----------------------------------------------------------

-- Find sites within radius (km) of a point
CREATE OR REPLACE FUNCTION dooh.sites_within_radius(
    lat DECIMAL,
    lon DECIMAL,
    radius_km DECIMAL
)
RETURNS TABLE (
    site_id UUID,
    name VARCHAR,
    distance_km DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.site_id,
        s.name,
        ST_Distance(
            s.location::geography,
            ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography
        ) / 1000 as distance_km
    FROM dooh.sites s
    WHERE ST_DWithin(
        s.location::geography,
        ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography,
        radius_km * 1000
    )
    ORDER BY distance_km;
END;
$$ LANGUAGE plpgsql;

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION dooh.update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to sites table
DROP TRIGGER IF EXISTS sites_updated_at ON dooh.sites;
CREATE TRIGGER sites_updated_at
    BEFORE UPDATE ON dooh.sites
    FOR EACH ROW
    EXECUTE FUNCTION dooh.update_timestamp();
