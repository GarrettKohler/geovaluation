"""
Feature Selection Integration Module.

Provides unified interface for integrating feature selection techniques
with the site scoring training pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path

from .config import FeatureSelectionConfig, FeatureSelectionMethod
from .stochastic_gates import StochasticGates, STGWrapper
from .gradient_analyzer import GradientFeatureAnalyzer, EpochWiseFeatureEliminator
from .shap_select import apply_shap_select
from .tabnet_wrapper import TabNetWrapper, create_tabnet_model


class FeatureSelectionTrainer:
    """
    Unified trainer that integrates feature selection with model training.

    This class wraps the standard training loop and applies the configured
    feature selection technique. It handles:

    1. STG: Wraps model with stochastic gates, adds regularization loss
    2. LassoNet: Uses specialized LassoNet architecture with Hier-Prox
    3. SHAP-Select: Runs post-training feature elimination
    4. TabNet: Replaces MLP with TabNet architecture
    5. Gradient Analysis: Monitors weight/gradient profiles during training

    Usage:
        fs_trainer = FeatureSelectionTrainer(config, model, feature_names)
        for epoch in epochs:
            train_loss, fs_loss = fs_trainer.train_step(batch, optimizer)
            fs_trainer.on_epoch_end(epoch)

        summary = fs_trainer.get_selection_summary()
    """

    def __init__(
        self,
        config: FeatureSelectionConfig,
        model: nn.Module,
        feature_names: List[str],
        input_dim: int,
        device: str = 'cpu',
    ):
        """
        Initialize feature selection trainer.

        Args:
            config: Feature selection configuration
            model: The base neural network model
            feature_names: List of all feature names
            input_dim: Total input dimension (after embedding)
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        self.config = config
        self.base_model = model
        self.feature_names = feature_names
        self.input_dim = input_dim
        self.device = device

        # The model to actually train (may be wrapped)
        self.model = model

        # Feature selection components
        self.stg_gates: Optional[StochasticGates] = None
        self.gradient_analyzer: Optional[GradientFeatureAnalyzer] = None
        self.epoch_eliminator: Optional[EpochWiseFeatureEliminator] = None

        # Active feature mask (True = feature is active)
        self.active_mask = torch.ones(len(feature_names), dtype=torch.bool)

        # Selection history
        self.selection_history: List[Dict] = []

        # Initialize based on method
        self._initialize_method()

    def _initialize_method(self):
        """Initialize the selected feature selection method."""
        method = self.config.method

        if method == FeatureSelectionMethod.STOCHASTIC_GATES:
            # Wrap model with stochastic gates
            self.stg_gates = StochasticGates(
                n_features=self.input_dim,
                sigma=self.config.stg_sigma,
                reg_weight=self.config.stg_lambda,
                init_mean=self.config.stg_init_mean,
            ).to(self.device)

            print(f"[FeatureSelection] STG initialized: lambda={self.config.stg_lambda}, "
                  f"sigma={self.config.stg_sigma}")

        elif method == FeatureSelectionMethod.GRADIENT_ANALYSIS:
            # Initialize gradient analyzer
            self.gradient_analyzer = GradientFeatureAnalyzer(
                model=self.base_model,
                input_dim=self.input_dim,
                feature_names=self.feature_names,
                analysis_interval=self.config.gradient_analysis_interval,
            )

            self.epoch_eliminator = EpochWiseFeatureEliminator(
                analyzer=self.gradient_analyzer,
                elimination_interval=self.config.gradient_elimination_interval,
                elimination_percentile=self.config.gradient_elimination_percentile,
                min_features=self.config.gradient_min_features,
            )

            print(f"[FeatureSelection] Gradient Analysis initialized: "
                  f"interval={self.config.gradient_analysis_interval}")

        elif method == FeatureSelectionMethod.TABNET:
            print("[FeatureSelection] TabNet mode - model should be TabNetWrapper")

        elif method == FeatureSelectionMethod.LASSONET:
            print("[FeatureSelection] LassoNet mode - model should be LassoNetModel")

        # Initialize gradient tracking if enabled (independent of method)
        if self.config.track_gradients and method != FeatureSelectionMethod.GRADIENT_ANALYSIS:
            self.gradient_analyzer = GradientFeatureAnalyzer(
                model=self.base_model,
                input_dim=self.input_dim,
                feature_names=self.feature_names,
                analysis_interval=self.config.gradient_analysis_interval,
            )
            print("[FeatureSelection] Gradient tracking enabled (monitoring only)")

    def apply_feature_gating(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply feature gating if using STG method.

        Args:
            x: Concatenated feature tensor (after embeddings)

        Returns:
            gated_x: Feature tensor with gating applied
            reg_loss: Regularization loss from gates
        """
        if self.stg_gates is not None:
            return self.stg_gates(x)
        return x, torch.tensor(0.0, device=self.device)

    def get_regularization_loss(self) -> torch.Tensor:
        """Get the regularization loss from feature selection."""
        if self.stg_gates is not None:
            return self.stg_gates.get_regularization_loss()
        return torch.tensor(0.0, device=self.device)

    def on_epoch_end(self, epoch: int) -> Dict:
        """
        Called at the end of each epoch.

        Performs epoch-wise operations like:
        - Recording gradient statistics
        - Checking for feature elimination
        - Updating selection history

        Returns:
            Dictionary with epoch statistics
        """
        stats = {'epoch': epoch}

        # Record gradients if tracking
        if self.gradient_analyzer is not None:
            self.gradient_analyzer.record_epoch(epoch)

        # Check for epoch-wise elimination (Gradient Analysis method)
        if self.epoch_eliminator is not None:
            eliminated = self.epoch_eliminator.check_elimination(epoch)
            if eliminated:
                stats['eliminated_features'] = eliminated
                self.active_mask = self.epoch_eliminator.get_active_feature_mask()
                print(f"[FeatureSelection] Epoch {epoch}: Eliminated {len(eliminated)} features: {eliminated}")

        # Record STG gate statistics
        if self.stg_gates is not None:
            gate_probs = self.stg_gates.get_gate_probabilities().cpu().numpy()
            n_active = (gate_probs > self.config.stg_threshold).sum()
            stats['n_active_features'] = int(n_active)
            stats['mean_gate_prob'] = float(gate_probs.mean())

            if epoch % 10 == 0:
                print(f"[FeatureSelection] Epoch {epoch}: {n_active}/{len(gate_probs)} features active, "
                      f"mean gate prob={gate_probs.mean():.3f}")

        self.selection_history.append(stats)
        return stats

    def get_selected_features(self, threshold: Optional[float] = None) -> List[str]:
        """Get list of currently selected (active) feature names."""
        threshold = threshold or self.config.selection_threshold

        if self.stg_gates is not None:
            return self.stg_gates.get_selected_feature_names(
                self.feature_names, threshold=self.config.stg_threshold
            )

        if self.epoch_eliminator is not None:
            return self.epoch_eliminator.get_active_features()

        # Return all features if no selection method active
        return self.feature_names

    def get_eliminated_features(self) -> List[str]:
        """Get list of eliminated feature names."""
        selected = set(self.get_selected_features())
        return [f for f in self.feature_names if f not in selected]

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from the current method."""
        if self.stg_gates is not None:
            probs = self.stg_gates.get_gate_probabilities().cpu().numpy()
            return {name: float(prob) for name, prob in zip(self.feature_names, probs)}

        if self.gradient_analyzer is not None:
            scores = self.gradient_analyzer.compute_importance_scores()
            return {name: scores[name]['combined_score_normalized'] for name in self.feature_names}

        return {name: 1.0 for name in self.feature_names}

    def get_selection_summary(self) -> Dict:
        """Get comprehensive summary of feature selection results."""
        importance = self.get_feature_importance()
        selected = self.get_selected_features()
        eliminated = self.get_eliminated_features()

        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        summary = {
            'method': self.config.get_method_display_name(),
            'n_total_features': len(self.feature_names),
            'n_selected': len(selected),
            'n_eliminated': len(eliminated),
            'selection_rate': len(selected) / len(self.feature_names),
            'selected_features': selected,
            'eliminated_features': eliminated,
            'importance_scores': importance,
            'top_10_features': [
                {'name': name, 'importance': imp}
                for name, imp in sorted_features[:10]
            ],
            'bottom_10_features': [
                {'name': name, 'importance': imp}
                for name, imp in sorted_features[-10:]
            ],
            'config': self.config.to_dict(),
        }

        # Add method-specific stats
        if self.stg_gates is not None:
            summary['stg_stats'] = {
                'mean_gate_prob': float(self.stg_gates.get_gate_probabilities().mean()),
                'n_fully_closed': int((self.stg_gates.get_gate_probabilities() < 0.01).sum()),
            }

        if self.gradient_analyzer is not None:
            summary['gradient_stats'] = self.gradient_analyzer.get_summary()

        return summary

    def run_shap_validation(
        self,
        model_predict_fn: Callable,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run SHAP-Select validation on the trained model.

        This is a post-training validation that uses SHAP values
        and statistical significance to identify features that
        the model actually uses.

        Args:
            model_predict_fn: Function that takes X and returns predictions
            X_val: Validation features
            y_val: Validation targets
            feature_names: Optional override for feature names

        Returns:
            SHAP-Select results dictionary
        """
        if not self.config.run_shap_validation:
            return {}

        feature_names = feature_names or self.feature_names

        print("[FeatureSelection] Running SHAP-Select validation...")
        return apply_shap_select(
            model_predict_fn=model_predict_fn,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_names,
            n_background=self.config.shap_n_background,
            n_shap_samples=self.config.shap_n_samples,
            significance_level=self.config.shap_significance_level,
            verbose=True,
        )

    def save_results(self, output_dir: Path):
        """Save feature selection results to disk."""
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = self.get_selection_summary()

        # Save summary JSON
        with open(output_dir / 'feature_selection_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save selection history
        with open(output_dir / 'feature_selection_history.json', 'w') as f:
            json.dump(self.selection_history, f, indent=2)

        # Save active feature mask
        np.save(output_dir / 'active_feature_mask.npy', self.active_mask.numpy())

        print(f"[FeatureSelection] Results saved to {output_dir}")


def create_feature_selection_model(
    config: FeatureSelectionConfig,
    base_model: nn.Module,
    n_numeric: int,
    n_boolean: int,
    categorical_vocab_sizes: Dict[str, int],
    embedding_dim: int = 16,
    hidden_dims: List[int] = None,
    dropout: float = 0.2,
    feature_names: List[str] = None,
    device: str = 'cpu',
) -> Tuple[nn.Module, Optional['FeatureSelectionTrainer']]:
    """
    Factory function to create a model with feature selection.

    Returns the appropriate model based on the configured method:
    - NONE: Returns base_model unchanged
    - STOCHASTIC_GATES: Returns base_model with STG wrapper
    - LASSONET: Returns new LassoNetModel
    - TABNET: Returns new TabNetWrapper
    - GRADIENT_ANALYSIS: Returns base_model with gradient tracking

    Args:
        config: Feature selection configuration
        base_model: The base SiteScoringModel
        n_numeric: Number of numeric features
        n_boolean: Number of boolean features
        categorical_vocab_sizes: Vocabulary sizes for categorical features
        embedding_dim: Embedding dimension
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate
        feature_names: List of feature names
        device: Device to use

    Returns:
        model: The model to train
        fs_trainer: FeatureSelectionTrainer instance (or None if no selection)
    """
    hidden_dims = hidden_dims or [512, 256, 128, 64]
    method = config.method

    # Calculate total input dimension (embedded categoricals + numeric + boolean)
    # Must match CategoricalEmbedding formula: max(min(embedding_dim, (vocab+1)//2), 4)
    cat_embedding_dim = sum(
        max(min(embedding_dim, (vocab_size + 1) // 2), 4)
        for vocab_size in categorical_vocab_sizes.values()
    )
    total_input_dim = cat_embedding_dim + n_numeric + n_boolean

    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(total_input_dim)]

    if method == FeatureSelectionMethod.NONE:
        # No feature selection - optionally still track gradients
        if config.track_gradients or config.run_shap_validation:
            fs_trainer = FeatureSelectionTrainer(
                config=config,
                model=base_model,
                feature_names=feature_names,
                input_dim=total_input_dim,
                device=device,
            )
            return base_model, fs_trainer
        return base_model, None

    elif method == FeatureSelectionMethod.STOCHASTIC_GATES:
        # Wrap with STG - requires modifying forward pass
        fs_trainer = FeatureSelectionTrainer(
            config=config,
            model=base_model,
            feature_names=feature_names,
            input_dim=total_input_dim,
            device=device,
        )
        return base_model, fs_trainer

    elif method == FeatureSelectionMethod.LASSONET:
        # Create LassoNet model (replaces base model)
        from .lassonet import LassoNetModel

        lassonet_model = LassoNetModel(
            n_numeric=n_numeric,
            n_boolean=n_boolean,
            categorical_vocab_sizes=categorical_vocab_sizes,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            M=config.lassonet_M,
        ).to(device)

        lassonet_model.set_feature_names(feature_names)

        fs_trainer = FeatureSelectionTrainer(
            config=config,
            model=lassonet_model,
            feature_names=feature_names,
            input_dim=total_input_dim,
            device=device,
        )
        return lassonet_model, fs_trainer

    elif method == FeatureSelectionMethod.TABNET:
        # Create TabNet wrapper (replaces base model)
        tabnet_model = create_tabnet_model(
            task_type='regression',
            n_d=config.tabnet_n_d,
            n_a=config.tabnet_n_a,
            n_steps=config.tabnet_n_steps,
            gamma=config.tabnet_gamma,
            mask_type=config.tabnet_mask_type,
            device=device,
        )

        tabnet_model.set_feature_names(feature_names)

        # TabNet handles its own feature selection
        return tabnet_model, None

    elif method == FeatureSelectionMethod.GRADIENT_ANALYSIS:
        # Use gradient analysis with base model
        fs_trainer = FeatureSelectionTrainer(
            config=config,
            model=base_model,
            feature_names=feature_names,
            input_dim=total_input_dim,
            device=device,
        )
        return base_model, fs_trainer

    else:
        raise ValueError(f"Unknown feature selection method: {method}")


class STGAugmentedModel(nn.Module):
    """
    Model wrapper that applies STG gating to the concatenated features
    before passing through the MLP.

    This keeps the original model architecture but adds learnable gates
    to the input features.
    """

    def __init__(
        self,
        base_model: nn.Module,
        stg_gates: StochasticGates,
    ):
        super().__init__()
        self.base_model = base_model
        self.stg_gates = stg_gates
        self._last_reg_loss = torch.tensor(0.0)

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        boolean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with STG gating.

        The gating is applied after feature concatenation, before MLP.
        """
        # Process categorical through embeddings
        cat_embedded = self.base_model.cat_embedding(categorical)

        # Normalize numeric features
        if self.base_model.numeric_bn is not None and self.base_model.n_numeric > 0:
            numeric = self.base_model.numeric_bn(numeric)

        # Concatenate all features
        x = torch.cat([cat_embedded, numeric, boolean], dim=1)

        # Apply STG gating
        x_gated, reg_loss = self.stg_gates(x)
        self._last_reg_loss = reg_loss

        # Pass through MLP
        x = self.base_model.mlp(x_gated)

        # Output
        return self.base_model.output(x)

    def get_regularization_loss(self) -> torch.Tensor:
        """Get the STG regularization loss from the last forward pass."""
        return self._last_reg_loss
