       Comprehensive Geospatial Codebase Exploration                                                                    
                                                                                                                                                                                                                       
       ---                                                                                                              
       PROJECT OVERVIEW                                                                                                 
                                                                                                                        
       This is a Geospatial Site Analysis & ML Scoring Platform for GSTV (appears to be related to out-of-home          
       advertising site analysis). The application provides:                                                            
       - Interactive WebGL-based map visualization of 57K+ sites across the US                                          
       - Real-time site filtering and revenue analysis                                                                  
       - PyTorch-based ML revenue prediction models optimized for Apple M4 silicon                                      
       - Advanced explainability features (SHAP, conformal prediction, counterfactuals)                                 
       - Feature selection capabilities (Stochastic Gates, LassoNet, SHAP-Select)                                       
                                                                                                                        
       ---                                                                                                              
       1. PROJECT STRUCTURE - ALL DIRECTORIES & KEY FILES                                                               
                                                                                                                        
       <project-root>/                                                                                    
       ├── app.py                                  # Flask web server entry point (25.9KB, 827 lines)                   
       ├── requirements.txt                        # Project dependencies                                               
       ├── README.md                               # Project documentation                                              
       ├── conformal_causality.md                  # Technical docs on causality                                        
       ├── ml_architecture_directions.md           # ML architecture notes                                              
       ├── knowledge-core.md                       # Institutional knowledge                                            
       ├── pytest.ini                              # Test configuration                                                 
       │                                                                                                                
       ├── src/                                    # Core library (services layer)                                      
       │   ├── __init__.py                                                                                              
       │   └── services/                                                                                                
       │       ├── __init__.py                                                                                          
       │       ├── data_service.py                 # Data loading & caching (502 lines)                                 
       │       ├── training_service.py             # ML training orchestration (1,316 lines)                            
       │       ├── nearest_site.py                 # KDTree spatial index for nearest sites (114 lines)                 
       │       ├── epa_walkability.py              # EPA walkability score lookups                                      
       │       ├── shap_service.py                 # SHAP feature importance computation (200+ lines)                   
       │       └── fleet_analysis_service.py       # Fleet-wide intervention analysis (150+ lines)                      
       │                                                                                                                
       ├── site_scoring/                           # ML Pipeline (PyTorch models, data processing)                      
       │   ├── __init__.py                                                                                              
       │   ├── config.py                           # Model configuration (148 lines)                                    
       │   ├── model.py                            # Neural network architecture (226 lines)                            
       │   ├── data_loader.py                      # Polars-based data processing (337 lines)                           
       │   ├── data_transform.py                   # Data transformation pipeline (765 lines)                           
       │   ├── train.py                            # Training loop (318 lines)                                          
       │   ├── predict.py                          # Inference module (186 lines)                                       
       │   ├── run.py                              # CLI entry point (85 lines)                                         
       │   │                                                                                                            
       │   ├── feature_selection/                  # Feature selection methods                                          
       │   │   ├── __init__.py                                                                                          
       │   │   ├── config.py                       # FS configuration                                                   
       │   │   ├── integration.py                  # FS integration pipeline                                            
       │   │   ├── stochastic_gates.py             # STG feature selection (11KB)                                       
       │   │   ├── lassonet.py                     # LassoNet feature selection (13KB)                                  
       │   │   ├── shap_select.py                  # SHAP-based feature selection (13KB)                                
       │   │   ├── tabnet_wrapper.py               # TabNet attention mechanism (12KB)                                  
       │   │   └── gradient_analyzer.py            # Gradient analysis for features (16KB)                              
       │   │                                                                                                            
       │   ├── explainability/                     # Explainability & calibration                                       
       │   │   ├── __init__.py                                                                                          
       │   │   ├── calibration.py                  # Probability calibration (isotonic/platt)                           
       │   │   ├── conformal.py                    # Conformal prediction (MAPIE)                                       
       │   │   ├── counterfactuals.py              # Counterfactual explanations (DICE)                                 
       │   │   ├── pipeline.py                     # Full explainability pipeline                                       
       │   │   └── tiers.py                        # Executive tier classification (Tier 1-4)                           
       │   │                                                                                                            
       │   └── outputs/                            # Model outputs & caching                                            
       │       ├── best_model.pt                   # Trained PyTorch checkpoint                                         
       │       ├── preprocessor.pkl                # Encoder/scaler state                                               
       │       ├── shap_cache.npz                  # Cached SHAP values                                                 
       │       ├── explainability/                 # Calibration & conformal components                                 
       │       ├── feature_selection_*.json        # Feature selection history                                          
       │       └── fleet_analysis/                 # Fleet analysis results                                             
       │                                                                                                                
       ├── templates/                              # Flask HTML templates                                               
       │   ├── index.html                          # Main map visualization (342KB)                                     
       │   ├── training_details.html               # Training UI (43KB)                                                 
       │   ├── shap_values.html                    # SHAP visualization (22KB)                                          
       │   └── glossary.html                       # ML/statistics glossary (41KB)                                      
       │                                                                                                                
       ├── data/                                   # Data storage (aggregated to single row/site)                       
       │   ├── input/                              # Raw CSV files                                                      
       │   │   ├── Sites - Base Data Set.csv       # Site coordinates & metadata                                        
       │   │   └── Site Scores - Site Revenue, Impressions, and Diagnostics.csv                                         
       │   │                                                                                                            
       │   ├── processed/                          # Cleaned & transformed data (for ML)                                
       │   │   ├── site_aggregated_precleaned.csv  # One row per site, pre-ML-cleaning (43MB)                           
       │   │   ├── site_aggregated_precleaned.parquet                                                                   
       │   │   ├── site_training_data.csv          # Final training dataset (20MB, 26K sites, 94 features)              
       │   │   ├── site_training_data.parquet      # Parquet version (5.9MB)                                            
       │   │   ├── precleaned_summary.txt          # Data summary                                                       
       │   │   └── training_data_summary.txt       # Feature breakdown report                                           
       │   │                                                                                                            
       │   ├── output/                             # Generated outputs (distances, etc.)                                
       │   │   ├── nearest_site_distances.csv                                                                           
       │   │   └── site_interstate_distances.csv                                                                        
       │   │                                                                                                            
       │   └── shapefiles/                         # Geographic boundary files                                          
       │       └── [TIGER census shapefiles]                                                                            
       │                                                                                                                
       ├── scripts/                                # Utility scripts                                                    
       │   ├── __init__.py                                                                                              
       │   └── site_walkability.py                 # EPA walkability score batch processor (29 lines)                   
       │                                                                                                                
       ├── tests/                                  # Test suite                                                         
       │   ├── __init__.py                                                                                              
       │   ├── conftest.py                                                                                              
       │   ├── test_api_sites.py                   # API endpoint tests                                                 
       │   ├── test_api_training.py                # Training API tests                                                 
       │   ├── test_data_service_unit.py           # Data service unit tests                                            
       │   ├── test_training_details_unit.py       # Training UI tests                                                  
       │   ├── test_training_service_unit.py       # Training service tests                                             
       │   ├── test_data_loading.py                # Data loader tests                                                  
       │   ├── test_frontend_data.py               # Frontend data tests                                                
       │   ├── test_revenue_consistency.py         # Revenue metric validation                                          
       │   └── test_regression.py                  # Regression tests                                                   
       │                                                                                                                
       ├── docs/                                   # Documentation                                                      
       │   ├── ARCHITECTURE.md                     # System architecture diagrams                                       
       │   ├── api.md                              # API reference                                                      
       │   ├── site_scoring/                       # ML module documentation                                            
       │   │   ├── model_neural_network_architecture.md                                                                 
       │   │   ├── data_loading_and_processing.md                                                                       
       │   │   ├── training_loop.md                                                                                     
       │   │   ├── predict_inference_engine.md                                                                          
       │   │   ├── config_site_scoring_module.md                                                                        
       │   │   └── init_site_scoring_ml_module.md                                                                       
       │   ├── src_docs/                                                                                                
       │   ├── scripts_docs/                                                                                            
       │   ├── secondary/                                                                                               
       │   ├── test_coverage.md                                                                                         
       │   └── conformal_causality_implementation_strategy.md                                                           
       │                                                                                                                
       └── static/                                 # CSS/JS assets                                                      
           └── [style assets]                                                                                           
                                                                                                                        
       ---                                                                                                              
       2. WHAT THE PROJECT DOES                                                                                         
                                                                                                                        
       Core Purpose                                                                                                     
                                                                                                                        
       Analysis and prediction of performance metrics for out-of-home advertising sites using geospatial data,          
       demographics, and machine learning.                                                                              
                                                                                                                        
       Primary Functions                                                                                                
                                                                                                                        
       A. Data Ingestion & Aggregation                                                                                  
       - Loads 1.47M monthly records from Site Scores CSV                                                               
       - Aggregates to 26,099 unique sites with 94 features per site                                                    
       - Joins geospatial data (Interstate distances, walkability scores)                                               
                                                                                                                        
       B. Web Visualization                                                                                             
       - Interactive WebGL map (Leaflet.js + Leaflet.glify) showing 57K+ sites                                          
       - Site filtering by 40+ categorical fields (State, Retailer, Network, etc.)                                      
       - Revenue coloring (white→green based on normalized revenue score)                                               
       - Lasso selection tool for multi-site selection                                                                  
       - Side panel with 8 categories of detailed site information                                                      
                                                                                                                        
       C. ML Model Training                                                                                             
       - Trains PyTorch neural network for revenue prediction (regression) or lookalike classification                  
       - Optimized for Apple M4 MPS GPU acceleration                                                                    
       - Supports configurable architectures: embedding layers + residual MLPs                                          
       - Features: numeric (12), categorical (9), boolean (40) features                                                 
       - Handles class imbalance for lookalike task (90/10 split)                                                       
                                                                                                                        
       D. Feature Selection & Explainability                                                                            
       - Feature Selection: Stochastic Gates (STG), LassoNet, SHAP-Select, TabNet                                       
       - Calibration: Isotonic regression for probability calibration                                                   
       - Conformal Prediction: Prediction sets with coverage guarantees (MAPIE/APS)                                     
       - Tier Classification: Maps probabilities to executive-friendly tiers (Recommended/Promising/Review/Not          
       Recommended)                                                                                                     
       - Counterfactuals: DICE-based explanations for intervention planning                                             
       - Fleet Analysis: Identifies strategic interventions across underperforming sites                                
                                                                                                                        
       ---                                                                                                              
       3. DATA CLEANING/PREPROCESSING CODE                                                                              
                                                                                                                        
       Main Pipeline: site_scoring/data_transform.py (765 lines)                                                        
                                                                                                                        
       # Key functions:                                                                                                 
       load_site_scores()                    # Load 1.47M monthly records                                               
       load_auxiliary_data()                 # Load geospatial distances                                                
       calculate_relative_strength()         # Momentum indicator (recent vs historical)                                
       calculate_all_relative_strength_features()  # RS for all metrics                                                 
                                                                                                                        
       Transformations Applied:                                                                                         
       1. Aggregates monthly data to single row per site                                                                
       2. Calculates total/average metrics across active months                                                         
       3. Filters to sites with >11 active months                                                                       
       4. Joins geospatial features (nearest site, interstate distance)                                                 
       5. Log-transforms 8 numeric columns (revenue, impressions, distances)                                            
       6. One-hot encodes 40 boolean flags (capabilities, restrictions)                                                 
       7. Calculates relative strength (momentum) for 4 metrics                                                         
                                                                                                                        
       Data Loader: site_scoring/data_loader.py (337 lines)                                                             
                                                                                                                        
       Uses Polars for fast CSV/Parquet parsing:                                                                        
                                                                                                                        
       class DataProcessor:                                                                                             
           _process_numeric()      # Scales, clips outliers (p1-p99)                                                    
           _process_categorical()  # Label encoding for embeddings                                                      
           _process_boolean()      # Already 0/1, passes through                                                        
           _process_target()       # Scales target variable                                                             
           load_and_process()      # Main entry point                                                                   
                                                                                                                        
       Data Cleaning Steps:                                                                                             
       - Null handling: fills NaN with 0                                                                                
       - Outlier clipping: constrains to p1-p99 percentiles                                                             
       - Feature scaling: StandardScaler (fitted on training set)                                                       
       - Categorical encoding: LabelEncoder (preserves ordering for embeddings)                                         
                                                                                                                        
       ---                                                                                                              
       4. DATA TRANSFORMATION CODE                                                                                      
                                                                                                                        
       Transformation Pipeline Overview                                                                                 
                                                                                                                        
       Input: 1.47M monthly Site Scores records                                                                         
       Output: 26,099 rows × 94 columns (one row per site)                                                              
                                                                                                                        
       Key Transformations (in data_transform.py):                                                                      
                                                                                                                        
       1. Temporal Aggregation                                                                                          
         - Groups by site (id_gbase)                                                                                    
         - Sums revenue/impressions across months                                                                       
         - Counts active months                                                                                         
       2. Relative Strength Calculation                                                                                 
         - Compares recent (30-day) to historical (90-day) performance                                                  
         - Formula: RS = (recent_avg + ε) / (historical_avg + ε)                                                        
         - Applied to: Impressions, NVIs, Revenue, Revenue/Screen                                                       
       3. Feature Engineering                                                                                           
         - Log transformations: log_total_revenue, log_nearest_site_distance_mi, etc.                                   
         - Calculates: revenue_per_screen, impressions_per_screen                                                       
         - Joins geospatial: Interstate distance, nearest site distance                                                 
       4. Encoding                                                                                                      
         - Boolean flags: Already 0/1 integers (no transformation)                                                      
         - Categoricals: Retained as strings (encoder handles later)                                                    
         - Numerics: Scaled during data loading                                                                         
       5. Filtering                                                                                                     
         - Removes sites with <12 active months                                                                         
         - Removes rows with missing key columns                                                                        
         - Result: 26,099 clean sites                                                                                   
                                                                                                                        
       Data Service Layer (src/services/data_service.py):                                                               
                                                                                                                        
       Handles in-memory caching:                                                                                       
       - load_sites(): Gets coordinates for all sites                                                                   
       - load_revenue_metrics(): Computes normalized revenue score (p10-p90 percentile normalization)                   
       - load_site_details(): Loads 52 detail columns per site                                                          
       - get_filter_options(): Returns unique values for categorical fields                                             
                                                                                                                        
       ---                                                                                                              
       5. MODEL TRAINING CODE                                                                                           
                                                                                                                        
       Training Orchestration (src/services/training_service.py, 1,316 lines)                                           
                                                                                                                        
       Core Classes:                                                                                                    
       - TrainingJob: Manages async training in background thread                                                       
       - TrainingConfig: User-configurable hyperparameters                                                              
       - TrainingProgress: Progress updates for SSE streaming                                                           
                                                                                                                        
       Training Flow:                                                                                                   
       1. User POSTs /api/training/start with config                                                                    
       2. Create TrainingJob, start in background thread                                                                
       3. Load data with Polars (fast parsing)                                                                          
       4. Create PyTorch model on device (MPS/CPU)                                                                      
       5. Initialize optional feature selection (STG/LassoNet/etc.)                                                     
       6. Train for N epochs with:                                                                                      
          - Task loss (Huber for regression, BCE+weight for classification)                                             
          - Feature selection regularization (if enabled)                                                               
          - Learning rate scheduling (ReduceLROnPlateau)                                                                
          - Early stopping (patience=10)                                                                                
       7. Save best model checkpoint                                                                                    
       8. Compute SHAP values for interpretability                                                                      
       9. Fit explainability pipeline (calibration + conformal)                                                         
       10. Stream all progress via SSE to frontend                                                                      
                                                                                                                        
       Apple M4 Optimizations:                                                                                          
                                                                                                                        
       - Detects chip with system_profiler                                                                              
       - Adjusts batch size, workers, prefetch based on chip tier                                                       
       - Uses torch.backends.mps.is_available() for GPU acceleration                                                    
       - SPMD (single program, multiple data) patterns                                                                  
                                                                                                                        
       Feature Selection Integration (Optional):                                                                        
                                                                                                                        
       if feature_selection_method != "none":                                                                           
           model = STGAugmentedModel(model, gates)  # Stochastic gates during training                                  
           fs_reg_loss = model.get_regularization_loss()  # L0 sparsity                                                 
                                                                                                                        
       Explainability Pipeline (For lookalike/classification):                                                          
                                                                                                                        
       # After training:                                                                                                
       calibrator = ProbabilityCalibrator(method='isotonic')                                                            
       calibrator.fit(y_proba_cal, y_cal)  # Calibrate on held-out set                                                  
                                                                                                                        
       conformal = ConformalClassifier(model, alpha=0.10)  # 90% coverage                                               
       conformal.fit(X_cal, y_cal)                                                                                      
                                                                                                                        
       tier_classifier = TierClassifier()  # Map to Tier 1-4                                                            
                                                                                                                        
       Model Architecture (site_scoring/model.py, 226 lines)                                                            
                                                                                                                        
       class SiteScoringModel(nn.Module):                                                                               
           def __init__(self,                                                                                           
               n_numeric: int,              # 12 continuous features                                                    
               n_boolean: int,              # 40 binary features                                                        
               categorical_vocab_sizes: Dict,  # 9 categorical features                                                 
               embedding_dim: int = 16,     # Embedding size per category                                               
               hidden_dims: List[int] = [512, 256, 128, 64],                                                            
               dropout: float = 0.2,                                                                                    
               use_batch_norm: bool = True):                                                                            
                                                                                                                        
               # Components:                                                                                            
               self.categorical_embeddings = CategoricalEmbedding(vocab_sizes, embedding_dim)                           
               self.numeric_bn = nn.BatchNorm1d(n_numeric)  # Normalize input numerics                                  
               self.residual_blocks = [ResidualBlock(...) for _ in hidden_dims]                                         
               self.output_layer = nn.Linear(hidden_dims[-1], 1)  # Regression/classification                           
                                                                                                                        
       Why This Architecture:                                                                                           
       - Embeddings for categoricals: High-cardinality features benefit from learned dense representations              
       - Residual connections: Enable training deeper networks without vanishing gradients                              
       - Batch normalization: Stabilizes large batch sizes on GPU                                                       
       - MPS-compatible: Uses only standard PyTorch ops (no unsupported CUDA kernels)                                   
                                                                                                                        
       ---                                                                                                              
       6. SERVICE LAYER CODE                                                                                            
                                                                                                                        
       src/services/data_service.py (502 lines)                                                                         
                                                                                                                        
       Loads and caches site data with singleton pattern                                                                
                                                                                                                        
       Key Functions:                                                                                                   
       load_sites()                    # Returns DataFrame: GTVID, Latitude, Longitude                                  
       load_revenue_metrics()          # Computes normalized revenue score per site                                     
       load_site_details()             # Gets 52 attributes per site (cached)                                           
       get_filter_options()            # Returns unique values for categorical fields                                   
       get_filtered_site_ids(filters)  # Filters sites by categorical criteria                                          
       get_site_details_for_display()  # Formats for side panel (8 categories)                                          
       preload_all_data()              # Warm cache on startup                                                          
       _clean_nan_values()             # Converts NaN/Inf to None for JSON                                              
                                                                                                                        
       Revenue Scoring (Normalization):                                                                                 
       1. Calculate revenue_per_day for each site                                                                       
       2. Get p10 and p90 percentiles across all sites                                                                  
       3. Normalize: (raw - p10) / (p90 - p10)                                                                          
       4. Clamp to [0, 1] range                                                                                         
                                                                                                                        
       Categorical Fields (40 filterable fields):                                                                       
       - Location: State, County, DMA                                                                                   
       - Site Info: Retailer, Network, Hardware, Experience, Program, Status                                            
       - Brands: Fuel Brand, Restaurant, C-Store                                                                        
       - Capabilities: EMV, NFC, Open 24h, Walk-up, Programmatic                                                        
       - Restrictions: 25 restriction flags (beer, wine, spirits, political, etc.)                                      
       - Sales: Sellable Site, Schedule Site                                                                            
                                                                                                                        
       src/services/training_service.py (1,316 lines)                                                                   
                                                                                                                        
       Manages ML model training with SSE progress streaming                                                            
                                                                                                                        
       Key Functions:                                                                                                   
       start_training(config)          # Start async training job                                                       
       stop_training()                 # Stop current job                                                               
       get_training_status()           # Get last known state                                                           
       stream_training_progress()      # SSE generator for live updates                                                 
       load_explainability_components()  # Load calibration/conformal                                                   
       explain_prediction()            # Single prediction explanation                                                  
       detect_apple_chip()             # Detect M1/M2/M3/M4 variant                                                     
       get_optimized_training_params() # Suggest batch size, workers                                                    
                                                                                                                        
       Apple Silicon Specs (hardcoded mappings):                                                                        
       - M1: 8 GPU cores, max batch 4096                                                                                
       - M1 Pro: 16 cores, max batch 8192                                                                               
       - M4 Pro: 20 cores, max batch 16384                                                                              
       - M4 Max: 40 cores, max batch 32768                                                                              
                                                                                                                        
       src/services/nearest_site.py (114 lines)                                                                         
                                                                                                                        
       Spatial KDTree index for nearest neighbor calculations                                                           
                                                                                                                        
       calculate_nearest_site_distances()                                                                               
           # Load sites, project to EPSG:5070 (NAD83 Conus Albers)                                                      
           # Build cKDTree spatial index                                                                                
           # Query for k=2 (self + nearest)                                                                             
           # Return distances in miles                                                                                  
                                                                                                                        
       Uses scipy.spatial.cKDTree for O(log n) queries.                                                                 
                                                                                                                        
       src/services/shap_service.py (200+ lines)                                                                        
                                                                                                                        
       SHAP feature importance computation and caching                                                                  
                                                                                                                        
       class ShapCache:                                                                                                 
           save()                  # Save SHAP values to npz                                                            
           load()                  # Load from cache                                                                    
           get_feature_importance()  # Get top N features by mean |SHAP|                                                
                                                                                                                        
       compute_shap_values()       # Main computation using KernelExplainer                                             
       generate_shap_plots()       # Create bar/summary plots as base64 PNG                                             
                                                                                                                        
       Uses shap.KernelExplainer (model-agnostic, works with PyTorch).                                                  
                                                                                                                        
       src/services/fleet_analysis_service.py (150+ lines)                                                              
                                                                                                                        
       Fleet-wide intervention analysis across low-performing sites                                                     
                                                                                                                        
       Key Classes:                                                                                                     
       @dataclass                                                                                                       
       class InterventionCluster:                                                                                       
           cluster_id: int                                                                                              
           name: str                           # E.g., "Extended Hours Initiative"                                      
           n_sites: int                        # Sites affected                                                         
           primary_changes: List[Dict]         # {feature, direction, magnitude}                                        
           estimated_tier_shift: Dict          # Expected improvement                                                   
           example_sites: List[str]            # Sample sites                                                           
                                                                                                                        
       @dataclass                                                                                                       
       class FleetAnalysisResult:                                                                                       
           interventions: List[InterventionCluster]                                                                     
           tier_distribution_before/after: Dict                                                                         
           site_counterfactuals: Dict[site_id, List[counterfactual]]                                                    
           site_cluster_assignments: Dict[site_id, cluster_id]                                                          
                                                                                                                        
       Process:                                                                                                         
       1. Filter to Tier 3-4 sites (low performers)                                                                     
       2. Generate counterfactuals for each site                                                                        
       3. Cluster counterfactuals by suggested changes                                                                  
       4. Estimate ROI per intervention                                                                                 
       5. Export to Excel with 4-sheet report                                                                           
                                                                                                                        
       ---                                                                                                              
       7. DATA FILES & FORMATS                                                                                          
                                                                                                                        
       Input Data (data/input/)                                                                                         
                                                                                                                        
       Sites - Base Data Set.csv (57K+ sites)                                                                           
       ├── Columns: GTVID, Latitude, Longitude, [other site metadata]                                                   
       └── One row per unique site                                                                                      
                                                                                                                        
       Site Scores - Site Revenue, Impressions, and Diagnostics.csv (1.47M rows)                                        
       ├── Columns: 50+ (gtvid, date, revenue, impressions, nvis, demographics, restrictions, capabilities)             
       └── One row per site-month observation                                                                           
                                                                                                                        
       Processed Data (data/processed/)                                                                                 
                                                                                                                        
       site_training_data.parquet (5.9MB, 26,099 rows × 94 columns)                                                     
       ├── Numeric: 12 features                                                                                         
       │   ├── rs_Impressions, rs_NVIs, rs_Revenue, rs_RevenuePerScreen (relative strength)                             
       │   ├── avg_monthly_revenue, log_total_revenue                                                                   
       │   ├── log_nearest_site_distance_mi, log_min_distance_to_interstate_mi                                          
       │   ├── avg_household_income, median_age, pct_female, pct_male                                                   
       │                                                                                                                
       ├── Categorical: 9 features                                                                                      
       │   ├── network, program, experience_type, hardware_type, retailer                                               
       │   ├── brand_fuel, brand_restaurant, brand_c_store                                                              
       │   └── nearest_interstate                                                                                       
       │                                                                                                                
       └── Boolean: 40 features                                                                                         
           ├── c_*_encoded (10 capability flags)                                                                        
           ├── r_*_encoded (30 restriction flags)                                                                       
           └── schedule_site_encoded, sellable_site_encoded                                                             
                                                                                                                        
       site_aggregated_precleaned.csv (43MB, human-readable version)                                                    
       └── Same as above but in CSV format                                                                              
                                                                                                                        
       training_data_summary.txt                                                                                        
       └── Feature breakdown, value ranges, null counts                                                                 
                                                                                                                        
       Generated Outputs (site_scoring/outputs/)                                                                        
                                                                                                                        
       best_model.pt                          # PyTorch checkpoint with weights + config                                
       preprocessor.pkl                       # Fitted scalers/encoders                                                 
       shap_cache.npz                         # Cached SHAP values                                                      
       feature_selection_history.json         # Feature selection results                                               
       feature_selection_summary.json         # Summary of selected features                                            
                                                                                                                        
       explainability/                                                                                                  
       ├── calibrator.pkl                     # Fitted isotonic regression                                              
       ├── conformal.pkl                      # MAPIE conformal predictor state                                         
       ├── tier_classifier.pkl                # TierClassifier configuration                                            
       └── metadata.pkl                       # Feature names, dimensions, etc.                                         
                                                                                                                        
       fleet_analysis/                                                                                                  
       └── [job_id]/                          # Results per analysis job                                                
           ├── interventions.json                                                                                       
           ├── site_assignments.json                                                                                    
           └── fleet_report_[timestamp].xlsx  # Excel export                                                            
                                                                                                                        
       ---                                                                                                              
       8. CONFIGURATION FILES                                                                                           
                                                                                                                        
       site_scoring/config.py (148 lines)                                                                               
                                                                                                                        
       Defines entire ML pipeline configuration                                                                         
                                                                                                                        
       @dataclass                                                                                                       
       class Config:                                                                                                    
           data_path = Path(".../site_training_data.parquet")                                                           
           output_dir = Path(".../site_scoring/outputs")                                                                
                                                                                                                        
           target = "avg_monthly_revenue"           # What to predict                                                   
           task_type = "regression"                 # or "lookalike"                                                    
           device = "mps" if torch.backends.mps.is_available() else "cpu"                                               
                                                                                                                        
           # Data                                                                                                       
           batch_size = 4096                        # M4 optimized                                                      
           num_workers = 4                                                                                              
           pin_memory = True                                                                                            
           train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15                                                         
                                                                                                                        
           # Features (12 numeric, 9 categorical, 40 boolean)                                                           
           numeric_features = ["rs_Impressions", "avg_monthly_revenue", ...]                                            
           categorical_features = ["network", "program", ...]                                                           
           boolean_features = ["c_emv_enabled_encoded", "r_lottery_encoded", ...]                                       
                                                                                                                        
           # Model                                                                                                      
           embedding_dim = 16                                                                                           
           hidden_dims = [512, 256, 128, 64]                                                                            
           dropout = 0.2                                                                                                
                                                                                                                        
           # Training                                                                                                   
           epochs = 50                                                                                                  
           learning_rate = 1e-4                                                                                         
           weight_decay = 1e-5                                                                                          
           early_stopping_patience = 10                                                                                 
                                                                                                                        
           # Feature Selection (optional)                                                                               
           feature_selection_method = "none"  # or "stg_light", "lassonet_standard", etc.                               
                                                                                                                        
           # Explainability (for lookalike tasks)                                                                       
           fit_explainability = True                                                                                    
           calibration_split = 0.5                                                                                      
           conformal_alpha = 0.10                                                                                       
                                                                                                                        
       Flask Config (in app.py):                                                                                        
                                                                                                                        
       SHAP_OUTPUT_DIR = DEFAULT_OUTPUT_DIR  # Imported from site_scoring.config                                      
       app.run(debug=True, port=8080)                                                                                   
                                                                                                                        
       ---                                                                                                              
       9. FLASK WEB APPLICATION (app.py, 827 lines)                                                                     
                                                                                                                        
       Page Routes                                                                                                      
                                                                                                                        
       GET  /                              # Main map visualization                                                     
       GET  /training-details              # Training UI with site records                                              
       GET  /glossary                       # ML/statistics glossary                                                    
       GET  /shap-values                    # SHAP feature importance page                                              
                                                                                                                        
       API Routes - Sites Data                                                                                          
                                                                                                                        
       GET  /api/sites                      # All sites with coordinates & revenue                                      
       GET  /api/site-details/<site_id>     # Comprehensive site details (8 categories)                                 
       POST /api/bulk-site-details          # Bulk details for multiple sites                                           
                                                                                                                        
       Returns JSON with:                                                                                               
       - Site metadata (retailer, network, hardware)                                                                    
       - Location (state, county, DMA, coordinates)                                                                     
       - Revenue (avg monthly, total, active months, score)                                                             
       - Demographics (income, age, ethnicity %)                                                                        
       - Performance (impressions, visits, latency)                                                                     
       - Capabilities (EMV, NFC, 24h, walk-up)                                                                          
       - Restrictions (40+ advertising restriction flags)                                                               
       - Sales (sellable, schedule status)                                                                              
                                                                                                                        
       API Routes - Filtering                                                                                           
                                                                                                                        
       GET  /api/filter-options             # Returns unique values for all categorical fields                          
       POST /api/filtered-sites             # Gets site IDs matching filters                                            
                                                                                                                        
       Request: {"filters": {"State": "TX", "Network": "Gilbarco"}}                                                     
       Response: {"site_ids": [...], "count": 1234}                                                                     
                                                                                                                        
       40 filterable fields covering location, site info, brands, capabilities, restrictions, sales.                    
                                                                                                                        
       API Routes - Model Training                                                                                      
                                                                                                                        
       GET  /api/training/system-info       # PyTorch version, GPU availability                                         
       GET  /api/training/features          # Feature lists currently configured                                        
       POST /api/training/start             # Start training job                                                        
       POST /api/training/stop              # Stop training                                                             
       GET  /api/training/status            # Current job status                                                        
       GET  /api/training/stream            # SSE stream for real-time progress                                         
                                                                                                                        
       API Routes - SHAP Feature Importance                                                                             
                                                                                                                        
       GET  /api/shap/available             # Check if SHAP data exists                                                 
       GET  /api/shap/summary               # Top N features by importance                                              
       GET  /api/shap/plots                 # SHAP visualizations as base64 PNG                                         
                                                                                                                        
       API Routes - Explainability                                                                                      
                                                                                                                        
       GET  /api/explainability/available           # Check if pipeline fitted                                          
       POST /api/explainability/explain             # Single prediction explanation                                     
       POST /api/explainability/explain-batch       # Batch explanations                                                
       POST /api/explainability/tier-summary        # Tier distribution                                                 
       POST /api/explainability/fleet-analysis      # Start fleet-wide analysis                                         
       GET  /api/explainability/fleet-analysis/status/<job_id>                                                          
       GET  /api/explainability/export-report/<job_id>                                                                  
                                                                                                                        
       ---                                                                                                              
       10. TEMPLATES & UI STRUCTURE                                                                                     
                                                                                                                        
       templates/index.html (342KB - Main application)                                                                  
                                                                                                                        
       Single Page Application with:                                                                                    
       - Leaflet.js map with WebGL rendering (57K+ points)                                                              
       - Lasso selection tool                                                                                           
       - Click-to-select individual sites                                                                               
       - Right-side panel with site details                                                                             
       - Filter interface (40 categorical fields)                                                                       
       - Search dropdown for site selection                                                                             
       - Revenue color scale visualization                                                                              
       - Zoom-based marker scaling                                                                                      
                                                                                                                        
       Libraries:                                                                                                       
       - Leaflet.js, Leaflet.glify (map + WebGL)                                                                        
       - Select2 (multi-select dropdown)                                                                                
       - jQuery 3.7.1                                                                                                   
                                                                                                                        
       templates/training_details.html (43KB)                                                                           
                                                                                                                        
       Training interface with:                                                                                         
       - System info display (GPU, PyTorch version)                                                                     
       - Training parameter configuration                                                                               
       - Progress bar and metrics display                                                                               
       - Loss curves, MAE, SMAPE, RMSE, R² tracking                                                                     
       - Feature selection status (active features count)                                                               
       - Real-time updates via SSE                                                                                      
                                                                                                                        
       templates/shap_values.html (22KB)                                                                                
                                                                                                                        
       SHAP visualization page:                                                                                         
       - Feature importance bar plot                                                                                    
       - SHAP summary plot                                                                                              
       - Top N features ranked by |mean SHAP value|                                                                     
       - Feature statistics (mean, std of SHAP)                                                                         
                                                                                                                        
       templates/glossary.html (41KB)                                                                                   
                                                                                                                        
       Educational reference:                                                                                           
       - ML/statistics terms explained                                                                                  
       - Revenue metrics definitions                                                                                    
       - Feature descriptions                                                                                           
       - Model architecture overview                                                                                    
                                                                                                                        
       ---                                                                                                              
       11. FEATURE SELECTION MODULES                                                                                    
                                                                                                                        
       site_scoring/feature_selection/ Directory                                                                        
                                                                                                                        
       Purpose: Reduce 94 features to most important ones during/after training                                         
                                                                                                                        
       Methods Supported:                                                                                               
                                                                                                                        
       1. Stochastic Gates (STG) - stochastic_gates.py (11KB)                                                           
         - L0 regularization during training                                                                            
         - Learns binary gates per feature                                                                              
         - Sparse gradient flow                                                                                         
         - Parameters: stg_lambda (sparsity), stg_sigma (gate spread)                                                   
       2. LassoNet - lassonet.py (13KB)                                                                                 
         - Feature selection through learned skip connections                                                           
         - L1 regularization on feature paths                                                                           
         - Solves entire regularization path                                                                            
       3. SHAP-Select - shap_select.py (13KB)                                                                           
         - Post-training feature selection                                                                              
         - Uses SHAP values to rank importance                                                                          
         - Cumulative importance threshold (e.g., 90%)                                                                  
       4. TabNet Wrapper - tabnet_wrapper.py (12KB)                                                                     
         - Attention-based feature selection                                                                            
         - Sequential feature masking                                                                                   
       5. Integration - integration.py (19KB)                                                                           
         - Orchestrates feature selection in training pipeline                                                          
         - Presets: stg_light, stg_aggressive, lassonet_standard, hybrid_stg_shap                                       
       6. Gradient Analysis - gradient_analyzer.py (16KB)                                                               
         - Tracks gradient magnitude per feature                                                                        
         - Identifies dead neurons/features                                                                             
                                                                                                                        
       ---                                                                                                              
       12. EXPLAINABILITY MODULES                                                                                       
                                                                                                                        
       site_scoring/explainability/ Directory                                                                           
                                                                                                                        
       Purpose: Make ML predictions transparent and actionable                                                          
                                                                                                                        
       Components:                                                                                                      
                                                                                                                        
       1. Calibration - calibration.py                                                                                  
         - Isotonic regression or Platt scaling                                                                         
         - Ensures predicted prob = actual frequency                                                                    
         - Metrics: Brier score before/after, Expected Calibration Error (ECE)                                          
       2. Conformal Prediction - conformal.py                                                                           
         - Prediction sets with coverage guarantees                                                                     
         - Method: APS (Adaptive Prediction Sets)                                                                       
         - Alpha = 0.10 → 90% target coverage                                                                           
         - Quantifies uncertainty                                                                                       
       3. Tier Classification - tiers.py                                                                                
         - Maps calibrated probability to Tier 1-4:                                                                     
             - Tier 1 (>85%): "Recommended" - Proceed to contract                                                       
           - Tier 2 (65-85%): "Promising" - Site visit required                                                         
           - Tier 3 (50-65%): "Review Required" - Detailed assessment                                                   
           - Tier 4 (<50%): "Not Recommended" - Do not pursue                                                           
         - Color-coded: Green/Yellow/Orange/Red                                                                         
       4. Counterfactuals - counterfactuals.py                                                                          
         - DICE-based explanations                                                                                      
         - "What if we change Feature X to Y?"                                                                          
         - Identifies minimal changes needed for tier upgrade                                                           
         - Clusters counterfactuals by intervention type                                                                
       5. Pipeline - pipeline.py                                                                                        
         - Orchestrates all components                                                                                  
         - Fitted automatically after training lookalike models                                                         
         - Saves state for later loading                                                                                
                                                                                                                        
       ---                                                                                                              
       SUMMARY TABLE                                                                                                    
       ┌───────────────────┬───────────────────────────┬───────┬──────────────────────────────────────────────────┐     
       │     Component     │          File(s)          │ Lines │                     Purpose                      │     
       ├───────────────────┼───────────────────────────┼───────┼──────────────────────────────────────────────────┤     
       │ Web App           │ app.py                    │ 827   │ Flask server, 20 API endpoints                   │     
       ├───────────────────┼───────────────────────────┼───────┼──────────────────────────────────────────────────┤     
       │ Data Service      │ data_service.py           │ 502   │ Load/cache site & revenue data                   │     
       ├───────────────────┼───────────────────────────┼───────┼──────────────────────────────────────────────────┤     
       │ Training          │ training_service.py       │ 1,316 │ ML orchestration, SSE streaming                  │     
       ├───────────────────┼───────────────────────────┼───────┼──────────────────────────────────────────────────┤     
       │ Model             │ model.py                  │ 226   │ PyTorch architecture (embeddings + residual MLP) │     
       ├───────────────────┼───────────────────────────┼───────┼──────────────────────────────────────────────────┤     
       │ Data Loader       │ data_loader.py            │ 337   │ Polars-based CSV/Parquet loading, scaling        │     
       ├───────────────────┼───────────────────────────┼───────┼──────────────────────────────────────────────────┤     
       │ Data Transform    │ data_transform.py         │ 765   │ 1.47M rows → 26K rows, feature engineering       │     
       ├───────────────────┼───────────────────────────┼───────┼──────────────────────────────────────────────────┤     
       │ Nearest Site      │ nearest_site.py           │ 114   │ KDTree spatial index                             │     
       ├───────────────────┼───────────────────────────┼───────┼──────────────────────────────────────────────────┤     
       │ SHAP              │ shap_service.py           │ 200+  │ Feature importance computation                   │     
       ├───────────────────┼───────────────────────────┼───────┼──────────────────────────────────────────────────┤     
       │ Fleet Analysis    │ fleet_analysis_service.py │ 150+  │ Fleet-wide interventions                         │     
       ├───────────────────┼───────────────────────────┼───────┼──────────────────────────────────────────────────┤     
       │ Feature Selection │ feature_selection/        │ 90KB  │ STG, LassoNet, SHAP-Select, TabNet               │     
       ├───────────────────┼───────────────────────────┼───────┼──────────────────────────────────────────────────┤     
       │ Explainability    │ explainability/           │ 50KB  │ Calibration, conformal, counterfactuals, tiers   │     
       ├───────────────────┼───────────────────────────┼───────┼──────────────────────────────────────────────────┤     
       │ Config            │ config.py                 │ 148   │ ML pipeline configuration                        │     
       ├───────────────────┼───────────────────────────┼───────┼──────────────────────────────────────────────────┤     
       │ Tests             │ tests/                    │ 1000+ │ 8 test files covering all major components       │     
       └───────────────────┴───────────────────────────┴───────┴──────────────────────────────────────────────────┘     
       ---                                                                                                              
       KEY ARCHITECTURAL PATTERNS                                                                                       
                                                                                                                        
       1. Singleton Caching (data_service.py): Module-level caches for loaded data                                      
       2. Background Threading (training_service.py): Training in daemon thread, progress via queue                     
       3. Server-Sent Events (SSE): Real-time training updates to frontend                                              
       4. Feature Engineering Pipeline: Temporal aggregation → log transforms → encoding                                
       5. Device-Aware ML: Auto-detect M4 chip, optimize batch size/workers                                             
       6. Layered Architecture: Flask → Services → ML Pipeline → Data                                                   
                                                                                                                        
       This is a production-grade application for geospatial site analysis with sophisticated ML capabilities           
       including feature selection, calibration, and explainability designed specifically for Apple Silicon             
       optimization.       