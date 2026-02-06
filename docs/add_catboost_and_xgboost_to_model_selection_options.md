 Changes Required                                                                                                      
                                                                                                                        
  1. Install Dependencies (New)                                                                                         
                                                                                                                        
  pip install catboost xgboost                                                                                          
                                                                                                                        
  Add to requirements.txt:                                                                                              
  catboost>=1.2                                                                                                         
  xgboost>=2.0                                                                                                          
                                                                                                                        
  ---                                                                                                                   
  2. Create New Model Classes (site_scoring/model.py)                                                                   
                                                                                                                        
  Add wrapper classes that match the existing interface:                                                                
                                                                                                                        
  # New file or add to model.py                                                                                         
  from catboost import CatBoostRegressor, CatBoostClassifier                                                            
  from xgboost import XGBRegressor, XGBClassifier                                                                       
                                                                                                                        
  class CatBoostModel:                                                                                                  
      """CatBoost wrapper matching SiteScoringModel interface."""                                                       
                                                                                                                        
      def __init__(self, task_type="regression", categorical_features=None, **kwargs):                                  
          self.task_type = task_type                                                                                    
          self.categorical_features = categorical_features or []                                                        
                                                                                                                        
          if task_type == "regression":                                                                                 
              self.model = CatBoostRegressor(                                                                           
                  iterations=kwargs.get('epochs', 1000),                                                                
                  learning_rate=kwargs.get('learning_rate', 0.03),                                                      
                  depth=kwargs.get('depth', 6),                                                                         
                  cat_features=self.categorical_features,                                                               
                  verbose=100,                                                                                          
              )                                                                                                         
          else:                                                                                                         
              self.model = CatBoostClassifier(                                                                          
                  iterations=kwargs.get('epochs', 1000),                                                                
                  learning_rate=kwargs.get('learning_rate', 0.03),                                                      
                  depth=kwargs.get('depth', 6),                                                                         
                  cat_features=self.categorical_features,                                                               
                  verbose=100,                                                                                          
              )                                                                                                         
                                                                                                                        
      def fit(self, X_train, y_train, X_val=None, y_val=None):                                                          
          eval_set = (X_val, y_val) if X_val is not None else None                                                      
          self.model.fit(X_train, y_train, eval_set=eval_set)                                                           
          return self                                                                                                   
                                                                                                                        
      def predict(self, X):                                                                                             
          return self.model.predict(X)                                                                                  
                                                                                                                        
  class XGBoostModel:                                                                                                   
      """XGBoost wrapper matching similar interface."""                                                                 
      # Similar implementation...                                                                                       
                                                                                                                        
  ---                                                                                                                   
  3. Update Training Service (src/services/training_service.py)                                                         
                                                                                                                        
  Add model creation branching logic around line 418:                                                                   
                                                                                                                        
  # Current code (line 418):                                                                                            
  model = SiteScoringModel(...)                                                                                         
                                                                                                                        
  # Change to:                                                                                                          
  if self.config.model_type == "neural_network":                                                                        
      model = SiteScoringModel(                                                                                         
          n_numeric=processor.n_numeric_features,                                                                       
          n_boolean=processor.n_boolean_features,                                                                       
          categorical_vocab_sizes=processor.categorical_vocab_sizes,                                                    
          embedding_dim=pytorch_config.embedding_dim,                                                                   
          hidden_dims=pytorch_config.hidden_dims,                                                                       
          dropout=pytorch_config.dropout,                                                                               
      )                                                                                                                 
      # Continue with PyTorch training loop...                                                                          
                                                                                                                        
  elif self.config.model_type == "catboost":                                                                            
      model = CatBoostModel(                                                                                            
          task_type=self.config.task_type,                                                                              
          categorical_features=list(processor.categorical_vocab_sizes.keys()),                                          
          epochs=self.config.epochs,                                                                                    
          learning_rate=self.config.learning_rate,                                                                      
      )                                                                                                                 
      # Use sklearn-style fit() instead of PyTorch training loop                                                        
      X_train, y_train = self._prepare_tabular_data(train_loader)                                                       
      X_val, y_val = self._prepare_tabular_data(val_loader)                                                             
      model.fit(X_train, y_train, X_val, y_val)                                                                         
                                                                                                                        
  elif self.config.model_type == "xgboost":                                                                             
      model = XGBoostModel(...)                                                                                         
      # Similar sklearn-style training                                                                                  
                                                                                                                        
  ---                                                                                                                   
  4. Add Data Preparation Method (src/services/training_service.py)                                                     
                                                                                                                        
  CatBoost/XGBoost need flat numpy arrays, not PyTorch tensors:                                                         
                                                                                                                        
  def _prepare_tabular_data(self, data_loader):                                                                         
      """Convert PyTorch DataLoader to numpy arrays for sklearn-style models."""                                        
      all_numeric, all_categorical, all_boolean, all_targets = [], [], [], []                                           
                                                                                                                        
      for numeric, categorical, boolean, target in data_loader:                                                         
          all_numeric.append(numeric.numpy())                                                                           
          all_categorical.append(categorical.numpy())                                                                   
          all_boolean.append(boolean.numpy())                                                                           
          all_targets.append(target.numpy())                                                                            
                                                                                                                        
      X = np.hstack([                                                                                                   
          np.vstack(all_numeric),                                                                                       
          np.vstack(all_categorical),                                                                                   
          np.vstack(all_boolean)                                                                                        
      ])                                                                                                                
      y = np.vstack(all_targets).ravel()                                                                                
                                                                                                                        
      return X, y                                                                                                       
                                                                                                                        
  ---                                                                                                                   
  5. Update UI Dropdown (templates/index.html)                                                                          
                                                                                                                        
  Change line 3316-3318:                                                                                                
                                                                                                                        
  <select class="training-form-select" id="model-type">                                                                 
      <option value="neural_network" selected>Neural Network (PyTorch)</option>                                         
      <option value="catboost">CatBoost (Gradient Boosting)</option>                                                    
      <option value="xgboost">XGBoost (Gradient Boosting)</option>                                                      
  </select>                                                                                                             
                                                                                                                        
  ---                                                                                                                   
  6. Update UI Hint Text (JavaScript)                                                                                   
                                                                                                                        
  Add dynamic hint based on selection:                                                                                  
                                                                                                                        
  document.getElementById('model-type').addEventListener('change', function() {                                         
      const hints = {                                                                                                   
          'neural_network': 'Embedding + Residual MLP optimized for M4 GPU',                                            
          'catboost': 'Gradient boosting with native categorical support - often best for tabular data',                
          'xgboost': 'Fast gradient boosting - excellent baseline for tabular regression'                               
      };                                                                                                                
      document.querySelector('#model-type + .training-form-hint').textContent = hints[this.value];                      
  });                                                                                                                   
                                                                                                                        
  ---                                                                                                                   
  7. Conditionally Hide Neural Network Options                                                                          
                                                                                                                        
  When CatBoost/XGBoost selected, hide irrelevant options:                                                              
  - Hidden layers (not applicable)                                                                                      
  - Embedding dimension (CatBoost handles categoricals natively)                                                        
  - Dropout (not applicable)                                                                                            
                                                                                                                        
  document.getElementById('model-type').addEventListener('change', function() {                                         
      const isNeuralNet = this.value === 'neural_network';                                                              
      document.getElementById('hidden-layers').closest('.training-form-group').style.display =                          
          isNeuralNet ? 'block' : 'none';                                                                               
      document.getElementById('embedding-dim').closest('.training-form-group').style.display =                          
          isNeuralNet ? 'block' : 'none';                                                                               
      document.getElementById('dropout').closest('.training-form-group').style.display =                                
          isNeuralNet ? 'block' : 'none';                                                                               
  });                                                                                                                   
                                                                                                                        
  ---                                                                                                                   
  8. Update SHAP Integration                                                                                            
                                                                                                                        
  SHAP works natively with tree models - update src/services/shap_service.py:                                           
                                                                                                                        
  if isinstance(model, (CatBoostModel, XGBoostModel)):                                                                  
      # Use TreeExplainer (much faster than DeepExplainer)                                                              
      explainer = shap.TreeExplainer(model.model)                                                                       
  else:                                                                                                                 
      # Keep existing DeepExplainer for neural network                                                                  
      explainer = shap.DeepExplainer(model, background_data)                                                            
                                                                                                                        
  ---                                                                                                                   
  Summary Table                                                                                                         
  ┌──────────────────────────────────┬──────────────────────────────────────┬───────────┐                               
  │               File               │             Change Type              │  Effort   │                               
  ├──────────────────────────────────┼──────────────────────────────────────┼───────────┤                               
  │ requirements.txt                 │ Add dependencies                     │ 5 min     │                               
  ├──────────────────────────────────┼──────────────────────────────────────┼───────────┤                               
  │ site_scoring/model.py            │ Add CatBoost/XGBoost wrapper classes │ 1-2 hours │                               
  ├──────────────────────────────────┼──────────────────────────────────────┼───────────┤                               
  │ src/services/training_service.py │ Add model branching + data prep      │ 2-3 hours │                               
  ├──────────────────────────────────┼──────────────────────────────────────┼───────────┤                               
  │ templates/index.html             │ Update dropdown + dynamic UI         │ 30 min    │                               
  ├──────────────────────────────────┼──────────────────────────────────────┼───────────┤                               
  │ src/services/shap_service.py     │ Add TreeExplainer support            │ 30 min    │                               
  └──────────────────────────────────┴──────────────────────────────────────┴───────────┘                               
  Total estimated effort: 4-6 hours                                                                                     
                                                                                                                        
  ---                                                                                                                   
  Why This Would Be Valuable                                                                                            
                                                                                                                        
  ★ Insight ─────────────────────────────────────                                                                       
  CatBoost/XGBoost advantages for your data:                                                                            
  1. Native categorical handling - CatBoost processes categoricals without manual encoding, often outperforming         
  embeddings                                                                                                            
  2. Faster training - Gradient boosting is typically 5-10x faster than neural networks for tabular data                
  3. Better interpretability - Feature importance is built-in, SHAP is faster with TreeExplainer                        
  4. Strong baselines - Kagglers consistently find gradient boosting wins on structured/tabular data                    
  ─────────────────────────────────────────────────                                                                     
                                                                                                                        
  Would you like me to implement this?                                                                                  
                                                                                                                        
✻ Sautéed for 1m 9s                                                                                                     
                                                                                                                        
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
❯                                                                                                                       
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ⏵⏵ accept edits on (shift+Tab to cycle) · PR #1                                                         98809 tokens  
                                                                                      current: 2.1.25 · latest: 2.1.25  

