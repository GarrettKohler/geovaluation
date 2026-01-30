"""
Counterfactual Explanations for Site Scoring.

Generates "what-if" explanations showing what changes would flip a low-value
site to high-value. Uses DiCE (Diverse Counterfactual Explanations) with
explicit constraints on which features can be modified.

Key Concepts:
- Immutable Features: Cannot change (location, demographics)
- Actionable Features: Business can change (hours, equipment, capabilities)
- Monotonic Constraints: Some features can only increase (screen count)

Reference: Mothilal et al. (2020) - "DiCE: Diverse Counterfactual Explanations"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import pickle
import warnings

# Check if DiCE is available
try:
    import dice_ml
    from dice_ml import Dice
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False
    warnings.warn(
        "DiCE not installed. Install with: pip install dice-ml\n"
        "Counterfactual generation will be disabled."
    )

from sklearn.cluster import KMeans


# ============================================================================
# Feature Classification for DOOH Sites
# ============================================================================

# Features that CANNOT be changed (location, demographics, identity)
IMMUTABLE_FEATURES = [
    # Location/Demographics
    'avg_household_income',
    'median_age',
    'pct_female',
    'pct_male',
    'log_min_distance_to_nearest_site_mi',
    'log_min_distance_to_interstate_mi',
    'log_min_distance_to_kroger_mi',
    'log_min_distance_to_mcdonalds_mi',
    'nearest_interstate',

    # Site Identity
    'network',
    'program',
    'retailer',
    'brand_fuel',
    'brand_restaurant',
    'brand_c_store',

    # Restriction flags (typically contractual/legal)
    'r_lottery_encoded',
    'r_government_encoded',
    'r_age_restriction_encoded',
    'r_alcohol_encoded',
    'r_cannabis_encoded',
    'r_gambling_encoded',
    'r_political_encoded',
    'r_tobacco_encoded',
]

# Features that CAN be changed by business decisions
ACTIONABLE_FEATURES = [
    # Experience/Content (can change freely)
    'experience_type',
    'hardware_type',

    # Capabilities (boolean - can enable)
    'c_emv_enabled_encoded',
    'c_nfc_enabled_encoded',
    'c_open_24_hours_encoded',
    'c_vistar_programmatic_enabled_encoded',
    'c_walk_up_enabled_encoded',
]

# Default permitted ranges for monotonic features
DEFAULT_PERMITTED_RANGES = {}


@dataclass
class Counterfactual:
    """Single counterfactual explanation."""
    original_features: Dict[str, Any]
    counterfactual_features: Dict[str, Any]
    changes: Dict[str, Tuple[Any, Any]]  # feature -> (old_value, new_value)
    predicted_class: int
    predicted_probability: float

    def get_change_summary(self) -> str:
        """Human-readable summary of changes."""
        if not self.changes:
            return "No changes needed"

        lines = []
        for feature, (old, new) in self.changes.items():
            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                if new > old:
                    lines.append(f"  • Increase {feature}: {old} → {new}")
                else:
                    lines.append(f"  • Decrease {feature}: {old} → {new}")
            else:
                lines.append(f"  • Change {feature}: {old} → {new}")
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
            'example_sites': self.example_sites[:5],  # Limit examples
        }


class CounterfactualGenerator:
    """
    Generates counterfactual explanations using DiCE.

    For a low-value site, counterfactuals answer: "What minimal changes
    would upgrade this site to high-value?"

    Key Design Decisions:
    1. Use 'genetic' method for tree-agnostic optimization
    2. Explicit feature constraints (actionable vs immutable)
    3. Monotonic constraints for operational features
    4. Generate diverse counterfactuals (not just one)

    Args:
        model: Sklearn-compatible model (or wrapped PyTorch)
        train_data: DataFrame used for training (defines feature ranges)
        feature_names: List of feature names in order
        continuous_features: Features that are continuous (vs categorical)
        actionable_features: Features that can be modified (default: ACTIONABLE_FEATURES)
        immutable_features: Features that cannot be modified (default: IMMUTABLE_FEATURES)
    """

    def __init__(
        self,
        model: Any,
        train_data: pd.DataFrame,
        feature_names: List[str],
        continuous_features: List[str],
        outcome_name: str = 'target',
        actionable_features: Optional[List[str]] = None,
        immutable_features: Optional[List[str]] = None,
    ):
        if not DICE_AVAILABLE:
            raise RuntimeError(
                "DiCE is required for counterfactual generation. "
                "Install with: pip install dice-ml"
            )

        self.feature_names = feature_names
        self.outcome_name = outcome_name

        # Set feature classification
        self.actionable_features = actionable_features or [
            f for f in ACTIONABLE_FEATURES if f in feature_names
        ]
        self.immutable_features = immutable_features or [
            f for f in IMMUTABLE_FEATURES if f in feature_names
        ]

        # Ensure train_data has the outcome column
        if outcome_name not in train_data.columns:
            # Add dummy outcome for DiCE initialization
            train_data = train_data.copy()
            train_data[outcome_name] = 0

        # Initialize DiCE data interface
        self.dice_data = dice_ml.Data(
            dataframe=train_data,
            continuous_features=[f for f in continuous_features if f in train_data.columns],
            outcome_name=outcome_name,
        )

        # Initialize DiCE model interface
        self.dice_model = dice_ml.Model(model=model, backend='sklearn')

        # Create DiCE explainer with genetic method
        # Genetic works for any model (tree, neural net, etc.)
        self.dice_exp = Dice(
            self.dice_data,
            self.dice_model,
            method='genetic',
        )

        # Store feature ranges for permitted_range constraints
        self._compute_feature_ranges(train_data)

    def _compute_feature_ranges(self, train_data: pd.DataFrame) -> None:
        """Compute min/max ranges for each feature."""
        self.feature_ranges = {}
        for col in self.feature_names:
            if col in train_data.columns:
                col_data = train_data[col]
                if pd.api.types.is_numeric_dtype(col_data):
                    self.feature_ranges[col] = (
                        float(col_data.min()),
                        float(col_data.max()),
                    )

    def generate(
        self,
        site_features: pd.DataFrame,
        n_counterfactuals: int = 5,
        desired_class: int = 1,
        permitted_range: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> List[Counterfactual]:
        """
        Generate counterfactual explanations for a single site.

        Args:
            site_features: DataFrame with single row of features
            n_counterfactuals: Number of diverse counterfactuals to generate
            desired_class: Target class (1 = high-value)
            permitted_range: Override feature ranges {feature: (min, max)}

        Returns:
            List of Counterfactual objects
        """
        if len(site_features) != 1:
            raise ValueError(f"Expected single site, got {len(site_features)}")

        # Build permitted range with monotonic constraints
        final_permitted_range = permitted_range or {}

        for feature, range_fn in DEFAULT_PERMITTED_RANGES.items():
            if feature in self.actionable_features and feature in site_features.columns:
                current_val = float(site_features[feature].iloc[0])
                final_permitted_range[feature] = range_fn(current_val)

        try:
            # Generate counterfactuals
            cf_result = self.dice_exp.generate_counterfactuals(
                query_instances=site_features[self.feature_names],
                total_CFs=n_counterfactuals,
                desired_class=desired_class,
                features_to_vary=self.actionable_features,
                permitted_range=final_permitted_range,
                proximity_weight=0.5,
                diversity_weight=1.0,
            )

            # Parse results
            return self._parse_counterfactuals(
                site_features,
                cf_result.cf_examples_list[0],
            )

        except Exception as e:
            warnings.warn(f"Counterfactual generation failed: {e}")
            return []

    def _parse_counterfactuals(
        self,
        original: pd.DataFrame,
        cf_examples: Any,
    ) -> List[Counterfactual]:
        """Parse DiCE output into Counterfactual objects."""
        counterfactuals = []

        cf_df = cf_examples.final_cfs_df
        if cf_df is None or len(cf_df) == 0:
            return counterfactuals

        original_dict = original[self.feature_names].iloc[0].to_dict()

        for _, cf_row in cf_df.iterrows():
            cf_dict = {f: cf_row[f] for f in self.feature_names if f in cf_row.index}

            # Identify changes
            changes = {}
            for feature in self.actionable_features:
                if feature in original_dict and feature in cf_dict:
                    old_val = original_dict[feature]
                    new_val = cf_dict[feature]
                    if old_val != new_val:
                        changes[feature] = (old_val, new_val)

            # Get prediction for counterfactual
            pred_class = int(cf_row.get(self.outcome_name, desired_class=1))
            pred_prob = cf_row.get('probability', 0.5)

            counterfactuals.append(Counterfactual(
                original_features=original_dict,
                counterfactual_features=cf_dict,
                changes=changes,
                predicted_class=pred_class,
                predicted_probability=float(pred_prob),
            ))

        return counterfactuals

    def generate_batch(
        self,
        sites: pd.DataFrame,
        n_counterfactuals: int = 3,
        progress_callback: Optional[callable] = None,
    ) -> Dict[int, List[Counterfactual]]:
        """
        Generate counterfactuals for multiple sites.

        Note: This can be slow (~500ms per site). Use sparingly for
        batch analysis.

        Args:
            sites: DataFrame with multiple sites
            n_counterfactuals: Number per site
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Dict mapping site index to list of counterfactuals
        """
        results = {}
        n_total = len(sites)

        for i, (idx, row) in enumerate(sites.iterrows()):
            site_df = pd.DataFrame([row])
            results[idx] = self.generate(site_df, n_counterfactuals)

            if progress_callback:
                progress_callback(i + 1, n_total)

        return results


class UpgradePathClusterer:
    """
    Clusters counterfactual changes to identify fleet-wide upgrade paths.

    Instead of individual site recommendations, this identifies patterns
    like "42 sites would upgrade by extending hours to 24/7" - enabling
    strategic investment decisions.

    Args:
        n_clusters: Number of upgrade path clusters to identify
        min_cluster_size: Minimum sites per cluster to be considered valid
    """

    def __init__(
        self,
        n_clusters: int = 5,
        min_cluster_size: int = 5,
    ):
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.kmeans = None
        self._fitted = False
        self.actionable_features: List[str] = []

    def fit(
        self,
        counterfactual_changes: Dict[int, List[Counterfactual]],
        actionable_features: List[str],
    ) -> 'UpgradePathClusterer':
        """
        Fit clustering on counterfactual change vectors.

        Args:
            counterfactual_changes: Output from CounterfactualGenerator.generate_batch
            actionable_features: Features that can be modified

        Returns:
            self for method chaining
        """
        self.actionable_features = actionable_features

        # Extract change vectors
        change_vectors = []
        site_indices = []

        for site_idx, cfs in counterfactual_changes.items():
            for cf in cfs:
                vector = self._changes_to_vector(cf.changes)
                if vector is not None:
                    change_vectors.append(vector)
                    site_indices.append(site_idx)

        if len(change_vectors) < self.n_clusters:
            warnings.warn(
                f"Not enough change vectors ({len(change_vectors)}) for "
                f"{self.n_clusters} clusters. Reducing cluster count."
            )
            self.n_clusters = max(1, len(change_vectors) // 2)

        if len(change_vectors) == 0:
            warnings.warn("No valid change vectors to cluster")
            self._fitted = False
            return self

        X = np.array(change_vectors)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(X)

        self._site_indices = np.array(site_indices)
        self._change_vectors = X
        self._fitted = True

        return self

    def _changes_to_vector(self, changes: Dict[str, Tuple[Any, Any]]) -> Optional[np.ndarray]:
        """Convert change dict to numeric vector."""
        vector = []
        for feature in self.actionable_features:
            if feature in changes:
                old, new = changes[feature]
                if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                    vector.append(new - old)
                else:
                    # Categorical change encoded as 1
                    vector.append(1.0 if old != new else 0.0)
            else:
                vector.append(0.0)

        if all(v == 0 for v in vector):
            return None  # No meaningful change

        return np.array(vector)

    def get_upgrade_paths(self, n_total_sites: int) -> List[UpgradePath]:
        """
        Get identified upgrade paths with business context.

        Args:
            n_total_sites: Total number of low-value sites (for percentage calc)

        Returns:
            List of UpgradePath objects describing strategic interventions
        """
        if not self._fitted:
            return []

        paths = []

        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            n_sites = mask.sum()

            if n_sites < self.min_cluster_size:
                continue

            # Get cluster center
            center = self.kmeans.cluster_centers_[cluster_id]

            # Identify primary changes
            primary_changes = self._interpret_cluster_center(center)

            # Get example site indices
            example_sites = self._site_indices[mask][:5].tolist()

            paths.append(UpgradePath(
                cluster_id=cluster_id,
                name=self._generate_path_name(primary_changes),
                description=self._generate_path_description(primary_changes),
                n_sites_applicable=int(n_sites),
                pct_of_portfolio=n_sites / n_total_sites if n_total_sites > 0 else 0,
                primary_changes=primary_changes,
                estimated_success_rate=0.75,  # Default - should be estimated
                example_sites=example_sites,
            ))

        # Sort by number of applicable sites
        paths.sort(key=lambda p: p.n_sites_applicable, reverse=True)
        return paths

    def _interpret_cluster_center(self, center: np.ndarray) -> List[str]:
        """Interpret cluster center as list of primary changes."""
        changes = []

        for i, (feature, value) in enumerate(zip(self.actionable_features, center)):
            if abs(value) > 0.1:  # Threshold for significant change
                if value > 0:
                    changes.append(f"Increase {feature}")
                else:
                    changes.append(f"Decrease {feature}")

        return changes[:3]  # Top 3 changes

    def _generate_path_name(self, changes: List[str]) -> str:
        """Generate short name for upgrade path."""
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

    def _generate_path_description(self, changes: List[str]) -> str:
        """Generate description for upgrade path."""
        if not changes:
            return "General operational improvements"

        return "Primary changes: " + "; ".join(changes)

    def save(self, path: Path) -> None:
        """Save clusterer state."""
        with open(path, 'wb') as f:
            pickle.dump({
                'n_clusters': self.n_clusters,
                'min_cluster_size': self.min_cluster_size,
                'kmeans': self.kmeans,
                'fitted': self._fitted,
                'actionable_features': self.actionable_features,
            }, f)

    @classmethod
    def load(cls, path: Path) -> 'UpgradePathClusterer':
        """Load clusterer from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(
            n_clusters=data['n_clusters'],
            min_cluster_size=data['min_cluster_size'],
        )
        instance.kmeans = data['kmeans']
        instance._fitted = data['fitted']
        instance.actionable_features = data['actionable_features']
        return instance
