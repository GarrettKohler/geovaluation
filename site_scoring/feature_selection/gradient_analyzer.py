"""
Gradient-based Feature Importance Analysis.

Implements the methodology from:
"Importance estimate of features via analysis of their weight and gradient profile"
Scientific Reports, October 2024
https://www.nature.com/articles/s41598-024-72640-4

Key idea: Analyze how weights and gradients change throughout training to infer
feature importance. Features whose associated weights converge quickly and
remain stable are important; features with erratic or diminishing weights
can be eliminated.

This is a MONITORING technique that can be applied to any neural network
during training without architectural modifications.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class FeatureStats:
    """Statistics for a single feature tracked across epochs."""
    weight_history: List[float] = field(default_factory=list)
    gradient_history: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)

    def add(self, epoch: int, weight_magnitude: float, gradient_magnitude: float):
        self.epochs.append(epoch)
        self.weight_history.append(weight_magnitude)
        self.gradient_history.append(gradient_magnitude)


class GradientFeatureAnalyzer:
    """
    Analyze weight and gradient profiles during training to identify
    non-contributing features.

    This analyzer hooks into the training loop and records:
    - Weight magnitudes for each input feature at each epoch
    - Gradient magnitudes during backpropagation

    From these trajectories, it computes importance metrics:
    - final_weight: Weight magnitude at end of training (higher = more important)
    - weight_stability: Inverse of weight variance (stable = important)
    - gradient_convergence: Low final gradient = feature has converged
    - combined_score: Weighted combination of all metrics

    Usage:
        analyzer = GradientFeatureAnalyzer(model, input_dim, feature_names)

        for epoch in range(n_epochs):
            # ... training ...
            analyzer.record_epoch(epoch)

        # After training
        scores = analyzer.compute_importance_scores()
        to_eliminate = analyzer.get_features_to_eliminate(percentile=10)
    """

    def __init__(
        self,
        model: nn.Module,
        input_dim: int,
        feature_names: Optional[List[str]] = None,
        analysis_interval: int = 1,
        first_layer_name: Optional[str] = None,
    ):
        """
        Initialize the analyzer.

        Args:
            model: Neural network model
            input_dim: Number of input features
            feature_names: Optional list of feature names
            analysis_interval: Record every N epochs
            first_layer_name: Name of the first layer (auto-detected if None)
        """
        self.model = model
        self.input_dim = input_dim
        self.feature_names = feature_names or [f"feature_{i}" for i in range(input_dim)]
        self.analysis_interval = analysis_interval
        self.first_layer_name = first_layer_name

        # Storage for feature statistics
        self.feature_stats: Dict[int, FeatureStats] = {
            i: FeatureStats() for i in range(input_dim)
        }

        # Track global training metrics
        self.epochs_recorded: List[int] = []
        self.total_gradient_norms: List[float] = []

        # Find first layer
        self._first_layer = self._find_first_layer()
        if self._first_layer is None:
            raise ValueError("Could not find first linear layer in model")

    def _find_first_layer(self) -> Optional[nn.Linear]:
        """Find the first Linear layer in the model."""
        if self.first_layer_name:
            for name, module in self.model.named_modules():
                if name == self.first_layer_name and isinstance(module, nn.Linear):
                    return module
            raise ValueError(f"Layer '{self.first_layer_name}' not found")

        # Auto-detect: find first Linear layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                return module

        return None

    def record_epoch(self, epoch: int):
        """
        Record weight magnitudes after an epoch.

        Call this at the end of each training epoch.
        """
        if epoch % self.analysis_interval != 0:
            return

        self.epochs_recorded.append(epoch)

        layer = self._first_layer
        if layer is None:
            return

        with torch.no_grad():
            W = layer.weight.data  # Shape: [hidden_dim, input_dim]

            # Record weight magnitude for each input feature
            for j in range(min(W.shape[1], self.input_dim)):
                # Feature importance = L2 norm of weights connected to this feature
                weight_magnitude = torch.norm(W[:, j]).item()

                # Gradient magnitude (if available)
                grad_magnitude = 0.0
                if layer.weight.grad is not None:
                    grad_magnitude = torch.norm(layer.weight.grad[:, j]).item()

                self.feature_stats[j].add(epoch, weight_magnitude, grad_magnitude)

    def record_gradients(self):
        """
        Record gradient magnitudes during training.

        Call this after loss.backward() but before optimizer.step().
        """
        layer = self._first_layer
        if layer is None or layer.weight.grad is None:
            return

        # Record total gradient norm
        total_norm = torch.norm(layer.weight.grad).item()
        self.total_gradient_norms.append(total_norm)

    def compute_importance_scores(self) -> Dict[str, Dict[str, float]]:
        """
        Compute feature importance scores based on weight/gradient trajectories.

        Returns:
            Dictionary mapping feature name to scores dict containing:
            - final_weight: Weight magnitude at end of training
            - mean_weight: Average weight magnitude across training
            - weight_stability: 1 / (weight_std + eps) - higher = more stable
            - gradient_convergence: 1 / (final_gradient + eps) - higher = converged
            - combined_score: Weighted combination of all metrics
        """
        scores = {}

        for j in range(self.input_dim):
            stats = self.feature_stats[j]
            name = self.feature_names[j]

            weight_vals = stats.weight_history
            grad_vals = stats.gradient_history

            if not weight_vals:
                scores[name] = {
                    'final_weight': 0.0,
                    'mean_weight': 0.0,
                    'weight_stability': 0.0,
                    'gradient_convergence': 0.0,
                    'combined_score': 0.0,
                }
                continue

            # Compute metrics
            final_weight = weight_vals[-1] if weight_vals else 0.0
            mean_weight = np.mean(weight_vals)
            weight_std = np.std(weight_vals) if len(weight_vals) > 1 else 0.0
            weight_stability = 1.0 / (weight_std + 1e-6)

            # Gradient convergence: low final gradient = feature has converged
            final_gradient = grad_vals[-1] if grad_vals else 1.0
            gradient_convergence = 1.0 / (final_gradient + 1e-6)

            # Combined score (weighted average, normalized later)
            combined = (
                0.4 * final_weight +
                0.3 * (weight_stability / 1000) +  # Scale stability
                0.2 * mean_weight +
                0.1 * min(gradient_convergence / 100, 1.0)  # Cap gradient contribution
            )

            scores[name] = {
                'final_weight': float(final_weight),
                'mean_weight': float(mean_weight),
                'weight_stability': float(weight_stability),
                'gradient_convergence': float(gradient_convergence),
                'combined_score': float(combined),
            }

        # Normalize combined scores to [0, 1]
        combined_values = [s['combined_score'] for s in scores.values()]
        max_combined = max(combined_values) if combined_values else 1.0
        min_combined = min(combined_values) if combined_values else 0.0
        range_combined = max_combined - min_combined

        if range_combined > 0:
            for name in scores:
                scores[name]['combined_score_normalized'] = (
                    (scores[name]['combined_score'] - min_combined) / range_combined
                )
        else:
            for name in scores:
                scores[name]['combined_score_normalized'] = 1.0

        return scores

    def get_feature_ranking(self) -> List[Tuple[str, float]]:
        """
        Get features ranked by importance (highest first).

        Returns:
            List of (feature_name, combined_score) tuples, sorted descending.
        """
        scores = self.compute_importance_scores()
        ranking = [
            (name, scores[name]['combined_score_normalized'])
            for name in self.feature_names
        ]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def get_features_to_eliminate(
        self,
        percentile: float = 10,
        min_eliminate: int = 0,
        max_eliminate: Optional[int] = None,
    ) -> List[str]:
        """
        Return features in the bottom percentile of importance.

        Args:
            percentile: Bottom X% of features to eliminate
            min_eliminate: Minimum number of features to eliminate
            max_eliminate: Maximum number of features to eliminate

        Returns:
            List of feature names to eliminate.
        """
        ranking = self.get_feature_ranking()

        # Calculate number to eliminate
        n_eliminate = max(
            min_eliminate,
            int(len(ranking) * percentile / 100)
        )

        if max_eliminate is not None:
            n_eliminate = min(n_eliminate, max_eliminate)

        # Bottom features (lowest scores)
        to_eliminate = [name for name, _ in ranking[-n_eliminate:]] if n_eliminate > 0 else []

        return to_eliminate

    def get_features_to_keep(self, percentile: float = 90) -> List[str]:
        """
        Return features in the top percentile of importance.

        Args:
            percentile: Top X% of features to keep

        Returns:
            List of feature names to keep.
        """
        ranking = self.get_feature_ranking()
        n_keep = int(len(ranking) * percentile / 100)
        n_keep = max(n_keep, 1)  # Keep at least one feature

        return [name for name, _ in ranking[:n_keep]]

    def get_summary(self) -> Dict:
        """
        Get comprehensive summary of the analysis.
        """
        scores = self.compute_importance_scores()
        ranking = self.get_feature_ranking()

        return {
            'n_features': self.input_dim,
            'n_epochs_recorded': len(self.epochs_recorded),
            'top_10_features': [
                {'name': name, 'score': score}
                for name, score in ranking[:10]
            ],
            'bottom_10_features': [
                {'name': name, 'score': score}
                for name, score in ranking[-10:]
            ],
            'full_scores': scores,
            'full_ranking': ranking,
        }

    def plot_weight_trajectories(
        self,
        features: Optional[List[str]] = None,
        top_n: int = 10,
    ) -> Optional['matplotlib.figure.Figure']:
        """
        Plot weight trajectories for specified features.

        Args:
            features: List of feature names to plot (default: top N)
            top_n: If features is None, plot top N features

        Returns:
            Matplotlib figure or None if matplotlib unavailable.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return None

        if features is None:
            ranking = self.get_feature_ranking()
            features = [name for name, _ in ranking[:top_n]]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Weight trajectories
        ax1 = axes[0]
        for name in features:
            idx = self.feature_names.index(name)
            stats = self.feature_stats[idx]
            if stats.epochs:
                ax1.plot(stats.epochs, stats.weight_history, label=name, alpha=0.7)

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Weight Magnitude (L2 norm)')
        ax1.set_title('Weight Trajectories During Training')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Gradient trajectories
        ax2 = axes[1]
        for name in features:
            idx = self.feature_names.index(name)
            stats = self.feature_stats[idx]
            if stats.epochs and stats.gradient_history:
                ax2.plot(stats.epochs, stats.gradient_history, label=name, alpha=0.7)

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Gradient Magnitude')
        ax2.set_title('Gradient Trajectories During Training')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class EpochWiseFeatureEliminator:
    """
    Wrapper that performs epoch-wise feature elimination during training.

    Integrates with GradientFeatureAnalyzer to periodically eliminate
    features that show low importance based on weight/gradient analysis.

    Args:
        analyzer: GradientFeatureAnalyzer instance
        elimination_interval: Eliminate features every N epochs
        elimination_percentile: Bottom X% to eliminate each interval
        min_features: Minimum number of features to keep
    """

    def __init__(
        self,
        analyzer: GradientFeatureAnalyzer,
        elimination_interval: int = 10,
        elimination_percentile: float = 5,
        min_features: int = 5,
    ):
        self.analyzer = analyzer
        self.elimination_interval = elimination_interval
        self.elimination_percentile = elimination_percentile
        self.min_features = min_features

        # Track eliminated features
        self.eliminated_features: List[str] = []
        self.elimination_epochs: List[int] = []

        # Active feature mask
        self.active_mask = torch.ones(analyzer.input_dim, dtype=torch.bool)

    def check_elimination(self, epoch: int) -> List[str]:
        """
        Check if features should be eliminated at this epoch.

        Returns list of newly eliminated features.
        """
        if epoch % self.elimination_interval != 0 or epoch == 0:
            return []

        n_active = self.active_mask.sum().item()
        if n_active <= self.min_features:
            return []

        # Get features to eliminate
        candidates = self.analyzer.get_features_to_eliminate(
            percentile=self.elimination_percentile,
            max_eliminate=max(1, int((n_active - self.min_features) * 0.1))
        )

        # Filter to only active features
        newly_eliminated = [
            f for f in candidates
            if f not in self.eliminated_features
        ]

        # Update tracking
        for feat in newly_eliminated:
            idx = self.analyzer.feature_names.index(feat)
            self.active_mask[idx] = False
            self.eliminated_features.append(feat)
            self.elimination_epochs.append(epoch)

        return newly_eliminated

    def get_active_feature_mask(self) -> torch.Tensor:
        """Return boolean mask of currently active features."""
        return self.active_mask

    def get_active_features(self) -> List[str]:
        """Return list of currently active feature names."""
        return [
            self.analyzer.feature_names[i]
            for i in range(self.analyzer.input_dim)
            if self.active_mask[i]
        ]

    def get_summary(self) -> Dict:
        """Get summary of elimination history."""
        return {
            'n_original_features': self.analyzer.input_dim,
            'n_active_features': self.active_mask.sum().item(),
            'n_eliminated': len(self.eliminated_features),
            'eliminated_features': self.eliminated_features,
            'elimination_epochs': self.elimination_epochs,
            'active_features': self.get_active_features(),
        }
