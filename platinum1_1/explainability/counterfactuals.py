"""
Counterfactual Explanations for Site Scoring.

Generates "what-if" explanations showing what changes would flip a low-value
site to high-value. Uses DiCE (Diverse Counterfactual Explanations).

Reference: Mothilal et al. (2020) - "DiCE: Diverse Counterfactual Explanations"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import pickle
import warnings

try:
    import dice_ml
    from dice_ml import Dice
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False
    warnings.warn("DiCE not installed. Counterfactual generation will be disabled.")

from sklearn.cluster import KMeans


# Features that CANNOT be changed (location, demographics, identity)
# NOTE: Kroger removed in platinum1_1 — not in data source
IMMUTABLE_FEATURES = [
    'avg_household_income', 'median_age', 'pct_female', 'pct_male',
    'log_min_distance_to_nearest_site_mi', 'log_min_distance_to_interstate_mi',
    'log_min_distance_to_mcdonalds_mi',
    'nearest_interstate',
    'network', 'program', 'retailer', 'brand_fuel', 'brand_c_store',
    'r_lottery_encoded', 'r_government_encoded',
]

# Features that CAN be changed by business decisions
ACTIONABLE_FEATURES = [
    'experience_type', 'hardware_type',
    'c_emv_enabled_encoded', 'c_nfc_enabled_encoded',
    'c_open_24_hours_encoded', 'c_vistar_programmatic_enabled_encoded',
]


@dataclass
class Counterfactual:
    """Single counterfactual explanation."""
    original_features: Dict[str, Any]
    counterfactual_features: Dict[str, Any]
    changes: Dict[str, Tuple[Any, Any]]
    predicted_class: int
    predicted_probability: float

    def get_change_summary(self) -> str:
        if not self.changes:
            return "No changes needed"
        lines = []
        for feature, (old, new) in self.changes.items():
            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                direction = "Increase" if new > old else "Decrease"
                lines.append(f"  * {direction} {feature}: {old} -> {new}")
            else:
                lines.append(f"  * Change {feature}: {old} -> {new}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            'original': self.original_features,
            'counterfactual': self.counterfactual_features,
            'changes': {k: list(v) for k, v in self.changes.items()},
            'predicted_class': self.predicted_class,
            'predicted_probability': self.predicted_probability,
        }


@dataclass
class UpgradePath:
    """Fleet-wide upgrade path identified from clustering counterfactuals."""
    cluster_id: int
    name: str
    description: str
    n_sites_applicable: int
    pct_of_portfolio: float
    primary_changes: List[str]
    estimated_success_rate: float
    example_sites: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'cluster_id': self.cluster_id,
            'name': self.name,
            'description': self.description,
            'n_sites': self.n_sites_applicable,
            'pct_portfolio': self.pct_of_portfolio,
            'primary_changes': self.primary_changes,
            'estimated_success_rate': self.estimated_success_rate,
            'example_sites': self.example_sites[:5],
        }


class CounterfactualGenerator:
    """Generates counterfactual explanations using DiCE."""

    def __init__(self, model, train_data, feature_names, continuous_features,
                 outcome_name='target', actionable_features=None, immutable_features=None):
        if not DICE_AVAILABLE:
            raise RuntimeError("DiCE is required. Install with: pip install dice-ml")

        self.feature_names = feature_names
        self.outcome_name = outcome_name

        self.actionable_features = actionable_features or [
            f for f in ACTIONABLE_FEATURES if f in feature_names
        ]
        self.immutable_features = immutable_features or [
            f for f in IMMUTABLE_FEATURES if f in feature_names
        ]

        if outcome_name not in train_data.columns:
            train_data = train_data.copy()
            train_data[outcome_name] = 0

        self.dice_data = dice_ml.Data(
            dataframe=train_data,
            continuous_features=[f for f in continuous_features if f in train_data.columns],
            outcome_name=outcome_name,
        )
        self.dice_model = dice_ml.Model(model=model, backend='sklearn')
        self.dice_exp = Dice(self.dice_data, self.dice_model, method='genetic')

    def generate(self, site_features, n_counterfactuals=5, desired_class=1,
                 permitted_range=None):
        """Generate counterfactual explanations for a single site."""
        if len(site_features) != 1:
            raise ValueError(f"Expected single site, got {len(site_features)}")

        try:
            cf_result = self.dice_exp.generate_counterfactuals(
                query_instances=site_features[self.feature_names],
                total_CFs=n_counterfactuals,
                desired_class=desired_class,
                features_to_vary=self.actionable_features,
                permitted_range=permitted_range or {},
                proximity_weight=0.5,
                diversity_weight=1.0,
            )
            return self._parse_counterfactuals(site_features, cf_result.cf_examples_list[0])
        except Exception as e:
            warnings.warn(f"Counterfactual generation failed: {e}")
            return []

    def _parse_counterfactuals(self, original, cf_examples):
        counterfactuals = []
        cf_df = cf_examples.final_cfs_df
        if cf_df is None or len(cf_df) == 0:
            return counterfactuals

        original_dict = original[self.feature_names].iloc[0].to_dict()

        for _, cf_row in cf_df.iterrows():
            cf_dict = {f: cf_row[f] for f in self.feature_names if f in cf_row.index}
            changes = {}
            for feature in self.actionable_features:
                if feature in original_dict and feature in cf_dict:
                    old_val = original_dict[feature]
                    new_val = cf_dict[feature]
                    if old_val != new_val:
                        changes[feature] = (old_val, new_val)

            pred_prob = cf_row.get('probability', 0.5)
            counterfactuals.append(Counterfactual(
                original_features=original_dict,
                counterfactual_features=cf_dict,
                changes=changes,
                predicted_class=1,
                predicted_probability=float(pred_prob),
            ))

        return counterfactuals

    def generate_batch(self, sites, n_counterfactuals=3, progress_callback=None):
        """Generate counterfactuals for multiple sites."""
        results = {}
        n_total = len(sites)
        for i, (idx, row) in enumerate(sites.iterrows()):
            site_df = pd.DataFrame([row])
            results[idx] = self.generate(site_df, n_counterfactuals)
            if progress_callback:
                progress_callback(i + 1, n_total)
        return results


class UpgradePathClusterer:
    """Clusters counterfactual changes to identify fleet-wide upgrade paths."""

    def __init__(self, n_clusters=5, min_cluster_size=5):
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.kmeans = None
        self._fitted = False
        self.actionable_features: List[str] = []

    def fit(self, counterfactual_changes, actionable_features):
        """Fit clustering on counterfactual change vectors."""
        self.actionable_features = actionable_features
        change_vectors, site_indices = [], []

        for site_idx, cfs in counterfactual_changes.items():
            for cf in cfs:
                vector = self._changes_to_vector(cf.changes)
                if vector is not None:
                    change_vectors.append(vector)
                    site_indices.append(site_idx)

        if len(change_vectors) < self.n_clusters:
            self.n_clusters = max(1, len(change_vectors) // 2)

        if len(change_vectors) == 0:
            self._fitted = False
            return self

        X = np.array(change_vectors)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(X)
        self._site_indices = np.array(site_indices)
        self._change_vectors = X
        self._fitted = True
        return self

    def _changes_to_vector(self, changes):
        vector = []
        for feature in self.actionable_features:
            if feature in changes:
                old, new = changes[feature]
                if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                    vector.append(new - old)
                else:
                    vector.append(1.0 if old != new else 0.0)
            else:
                vector.append(0.0)
        if all(v == 0 for v in vector):
            return None
        return np.array(vector)

    def get_upgrade_paths(self, n_total_sites):
        """Get identified upgrade paths with business context."""
        if not self._fitted:
            return []

        paths = []
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            n_sites = mask.sum()
            if n_sites < self.min_cluster_size:
                continue

            center = self.kmeans.cluster_centers_[cluster_id]
            primary_changes = self._interpret_cluster_center(center)
            example_sites = self._site_indices[mask][:5].tolist()

            paths.append(UpgradePath(
                cluster_id=cluster_id,
                name=self._generate_path_name(primary_changes),
                description=self._generate_path_description(primary_changes),
                n_sites_applicable=int(n_sites),
                pct_of_portfolio=n_sites / n_total_sites if n_total_sites > 0 else 0,
                primary_changes=primary_changes,
                estimated_success_rate=0.75,
                example_sites=example_sites,
            ))

        paths.sort(key=lambda p: p.n_sites_applicable, reverse=True)
        return paths

    def _interpret_cluster_center(self, center):
        changes = []
        for i, (feature, value) in enumerate(zip(self.actionable_features, center)):
            if abs(value) > 0.1:
                direction = "Increase" if value > 0 else "Decrease"
                changes.append(f"{direction} {feature}")
        return changes[:3]

    def _generate_path_name(self, changes):
        if not changes:
            return "General Optimization"
        first_change = changes[0].lower()
        if 'hours' in first_change or '24' in first_change:
            return "Extended Hours Initiative"
        elif 'screen' in first_change:
            return "Screen Expansion"
        elif 'programmatic' in first_change:
            return "Programmatic Enablement"
        elif 'emv' in first_change or 'nfc' in first_change:
            return "Payment Modernization"
        else:
            return "Capability Enhancement"

    def _generate_path_description(self, changes):
        if not changes:
            return "General operational improvements"
        return "Primary changes: " + "; ".join(changes)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'n_clusters': self.n_clusters,
                'min_cluster_size': self.min_cluster_size,
                'kmeans': self.kmeans,
                'fitted': self._fitted,
                'actionable_features': self.actionable_features,
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        instance = cls(n_clusters=data['n_clusters'], min_cluster_size=data['min_cluster_size'])
        instance.kmeans = data['kmeans']
        instance._fitted = data['fitted']
        instance.actionable_features = data['actionable_features']
        return instance
