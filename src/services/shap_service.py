"""
SHAP (SHapley Additive exPlanations) service for model interpretability.

Computes feature importance values after model training and caches results
alongside the model checkpoint for efficient retrieval.

Uses KernelExplainer for the PyTorch neural network (model-agnostic approach)
with a background sample for efficient Shapley value estimation.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import io
import base64


class ShapCache:
    """
    Manages storage and retrieval of SHAP computation results.
    Saves alongside model checkpoint in the outputs directory.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.cache_path = output_dir / "shap_cache.npz"

    def save(self, shap_values: np.ndarray, base_value: float,
             feature_names: List[str], sample_data: np.ndarray):
        """Save SHAP computation results to compressed numpy archive."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.cache_path,
            shap_values=shap_values,
            base_value=np.array([base_value]),
            feature_names=np.array(feature_names),
            sample_data=sample_data
        )

    def load(self) -> Optional[Dict]:
        """Load cached SHAP results. Returns None if no cache exists."""
        if not self.cache_path.exists():
            return None
        try:
            data = np.load(self.cache_path, allow_pickle=True)
            return {
                'shap_values': data['shap_values'],
                'base_value': float(data['base_value'][0]),
                'feature_names': data['feature_names'].tolist(),
                'sample_data': data['sample_data']
            }
        except Exception as e:
            print(f"Warning: Failed to load SHAP cache: {e}")
            return None

    def exists(self) -> bool:
        """Check if SHAP cache file exists."""
        return self.cache_path.exists()

    def get_feature_importance(self, top_n: int = 30) -> Optional[Dict]:
        """
        Get global feature importance as a JSON-serializable dict.
        Returns features ranked by mean |SHAP value|.
        """
        cached = self.load()
        if cached is None:
            return None

        shap_values = cached['shap_values']
        feature_names = cached['feature_names']

        # Mean absolute SHAP value per feature (global importance)
        importance = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(importance)[::-1][:top_n]

        features = []
        for i in sorted_idx:
            features.append({
                'feature': feature_names[i],
                'importance': float(importance[i]),
                'mean_shap': float(shap_values[:, i].mean()),
                'std_shap': float(shap_values[:, i].std()),
            })

        return {
            'features': features,
            'base_value': cached['base_value'],
            'n_samples': int(shap_values.shape[0]),
            'n_features': int(shap_values.shape[1]),
            'total_features_available': len(feature_names),
        }


def compute_shap_values(
    model: torch.nn.Module,
    processor,
    test_loader: torch.utils.data.DataLoader,
    feature_names: List[str],
    output_dir: Path,
    device: str = "cpu",
    n_background: int = 100,
    n_explain: int = 200,
    task_type: str = "regression",
    progress_callback=None,
) -> bool:
    """
    Compute SHAP values for the trained model using KernelExplainer.

    The model is moved to CPU for SHAP computation (MPS not fully supported
    by SHAP's gradient computations). Uses a subset of test data for efficiency.

    Args:
        model: Trained PyTorch model
        processor: DataProcessor with scaler info
        test_loader: DataLoader for test data
        feature_names: List of all feature names (numeric + categorical + boolean)
        output_dir: Directory to save SHAP cache
        device: Original training device (model will be moved to CPU)
        n_background: Number of background samples for KernelExplainer
        n_explain: Number of samples to explain
        task_type: "regression" or "lookalike"
        progress_callback: Optional function to report progress messages

    Returns:
        True if SHAP computation succeeded, False otherwise
    """
    try:
        import shap
    except ImportError:
        print("Warning: SHAP library not installed. Skipping SHAP computation.")
        if progress_callback:
            progress_callback("SHAP library not available - skipping feature importance")
        return False

    try:
        if progress_callback:
            progress_callback("Computing feature importance (SHAP)...")

        # Collect test data samples into numpy arrays
        all_numeric = []
        all_categorical = []
        all_boolean = []

        for numeric, categorical, boolean, _ in test_loader:
            all_numeric.append(numeric.numpy())
            all_categorical.append(categorical.numpy())
            all_boolean.append(boolean.numpy())
            # Collect enough samples
            total_so_far = sum(a.shape[0] for a in all_numeric)
            if total_so_far >= max(n_background, n_explain):
                break

        numeric_np = np.concatenate(all_numeric, axis=0)
        categorical_np = np.concatenate(all_categorical, axis=0)
        boolean_np = np.concatenate(all_boolean, axis=0)

        # Combine all features into a single 2D array for SHAP
        # This flattens the three input types into one feature matrix
        combined_data = np.concatenate([numeric_np, categorical_np, boolean_np], axis=1)

        # Select background and explanation samples
        n_total = combined_data.shape[0]
        bg_size = min(n_background, n_total // 2)
        explain_size = min(n_explain, n_total - bg_size)

        background_data = combined_data[:bg_size]
        explain_data = combined_data[bg_size:bg_size + explain_size]

        if progress_callback:
            progress_callback(f"SHAP: Using {bg_size} background, explaining {explain_size} samples...")

        # Move model to CPU for SHAP (MPS doesn't support all SHAP operations)
        model_cpu = model.cpu()
        model_cpu.eval()

        n_numeric = numeric_np.shape[1]
        n_categorical = categorical_np.shape[1]

        # Create prediction function that SHAP can call
        def predict_fn(X):
            """Wraps PyTorch model for SHAP's KernelExplainer."""
            with torch.no_grad():
                # Split combined features back into the three input types
                numeric_t = torch.FloatTensor(X[:, :n_numeric])
                categorical_t = torch.LongTensor(X[:, n_numeric:n_numeric + n_categorical].astype(int))
                boolean_t = torch.FloatTensor(X[:, n_numeric + n_categorical:])

                output = model_cpu(numeric_t, categorical_t, boolean_t)

                if task_type == "lookalike":
                    # Return probability for classification
                    return torch.sigmoid(output).numpy().flatten()
                else:
                    return output.numpy().flatten()

        # Use KernelExplainer (model-agnostic, works with any prediction function)
        explainer = shap.KernelExplainer(predict_fn, background_data)

        if progress_callback:
            progress_callback("SHAP: Computing Shapley values (this may take a moment)...")

        # Compute SHAP values with controlled number of evaluations
        shap_values = explainer.shap_values(explain_data, nsamples=150, silent=True)

        # Get base value (expected prediction on background)
        base_value = float(explainer.expected_value)

        # Save to cache
        cache = ShapCache(output_dir)
        cache.save(
            shap_values=shap_values,
            base_value=base_value,
            feature_names=feature_names,
            sample_data=explain_data
        )

        if progress_callback:
            progress_callback("SHAP: Feature importance computed successfully!")

        # Move model back to original device if needed
        if device != "cpu":
            model.to(torch.device(device))

        return True

    except Exception as e:
        print(f"Warning: SHAP computation failed: {e}")
        import traceback
        traceback.print_exc()
        if progress_callback:
            progress_callback(f"SHAP computation skipped: {str(e)[:80]}")
        # Move model back to original device
        if device != "cpu":
            try:
                model.to(torch.device(device))
            except:
                pass
        return False


def generate_shap_plots(output_dir: Path) -> Optional[Dict[str, str]]:
    """
    Generate SHAP visualization plots as base64-encoded PNG images.

    Returns dict with keys:
        - 'bar_plot': Feature importance bar chart
        - 'summary_plot': Beeswarm/dot summary plot

    Returns None if SHAP cache doesn't exist.
    """
    cache = ShapCache(output_dir)
    cached = cache.load()
    if cached is None:
        return None

    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for server
        import matplotlib.pyplot as plt

        shap_values = cached['shap_values']
        feature_names = cached['feature_names']
        sample_data = cached['sample_data']

        plots = {}

        # 1. Bar plot - Global feature importance
        fig, ax = plt.subplots(figsize=(10, 8))
        importance = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(importance)[::-1][:20]  # Top 20

        # Create horizontal bar chart
        y_pos = np.arange(len(sorted_idx))
        ax.barh(y_pos, importance[sorted_idx][::-1],
                color='#3bb0a5', edgecolor='#2a8a80', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx][::-1], fontsize=9)
        ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
        ax.set_title('Feature Importance (Top 20)', fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        plots['bar_plot'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # 2. Summary/beeswarm plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use SHAP's built-in summary plot
        # Create an Explanation object for the plot
        explanation = shap.Explanation(
            values=shap_values[:, sorted_idx[:20]],
            data=sample_data[:, sorted_idx[:20]],
            feature_names=[feature_names[i] for i in sorted_idx[:20]]
        )
        shap.plots.beeswarm(explanation, show=False, max_display=20)
        plt.title('SHAP Summary (Feature Impact Distribution)', fontsize=13, fontweight='bold')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        plots['summary_plot'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return plots

    except Exception as e:
        print(f"Warning: Failed to generate SHAP plots: {e}")
        import traceback
        traceback.print_exc()
        return None
