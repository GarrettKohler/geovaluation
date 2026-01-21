"""Site prioritization combining all three models."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ..config import Config, config as default_config
from ..models.similarity import SimilarityModel
from ..models.causal import CausalModel
from ..models.classifier import ActivationClassifier


@dataclass
class SiteRecommendation:
    """Recommendation for a single site."""

    site_id: str
    priority_rank: int
    priority_score: float
    lookalike_score: float
    success_probability: float
    expected_uplift: float
    uplift_confident: bool
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site_id": self.site_id,
            "priority_rank": self.priority_rank,
            "priority_score": self.priority_score,
            "lookalike_score": self.lookalike_score,
            "success_probability": self.success_probability,
            "expected_uplift": self.expected_uplift,
            "uplift_confident": self.uplift_confident,
            "recommended_action": self.recommended_action,
        }


class SitePrioritizer:
    """Combine all models to prioritize inactive sites.

    Scores sites on three dimensions:
    1. Lookalike similarity to high performers
    2. Predicted activation success probability
    3. Expected treatment uplift from hardware changes

    Produces ranked recommendations with confidence indicators.
    """

    def __init__(
        self,
        similarity_model: SimilarityModel,
        causal_model: CausalModel,
        classifier: ActivationClassifier,
        config: Optional[Config] = None,
    ):
        """Initialize prioritizer with fitted models.

        Args:
            similarity_model: Fitted SimilarityModel
            causal_model: Fitted CausalModel
            classifier: Fitted ActivationClassifier
            config: Configuration with priority weights
        """
        self.similarity_model = similarity_model
        self.causal_model = causal_model
        self.classifier = classifier
        self.config = config or default_config

    def prioritize(
        self,
        sites: pd.DataFrame,
        weight_lookalike: Optional[float] = None,
        weight_success: Optional[float] = None,
        weight_uplift: Optional[float] = None,
    ) -> pd.DataFrame:
        """Score and rank sites by priority.

        Args:
            sites: DataFrame with site_id and features
            weight_lookalike: Weight for lookalike score (default from config)
            weight_success: Weight for success probability (default from config)
            weight_uplift: Weight for expected uplift (default from config)

        Returns:
            DataFrame with scores and rankings, sorted by priority
        """
        weight_lookalike = weight_lookalike or self.config.model.priority_weight_lookalike
        weight_success = weight_success or self.config.model.priority_weight_success
        weight_uplift = weight_uplift or self.config.model.priority_weight_uplift

        scores = sites[["site_id"]].copy()

        # 1. Lookalike scores
        scores["lookalike_score"] = self.similarity_model.score(sites)

        # 2. Success probability from classifier
        scores["success_probability"] = self.classifier.predict_proba(sites)

        # 3. Expected uplift from causal model
        scores["expected_uplift"] = self.causal_model.effect(sites)
        lower, upper = self.causal_model.effect_interval(sites, alpha=0.05)
        scores["uplift_lower_bound"] = lower
        scores["uplift_upper_bound"] = upper
        scores["uplift_confident"] = lower > 0

        # Combined priority score using percentile ranks
        scores["priority_score"] = (
            weight_lookalike * scores["lookalike_score"].rank(pct=True)
            + weight_success * scores["success_probability"].rank(pct=True)
            + weight_uplift * scores["expected_uplift"].rank(pct=True)
        )

        # Rank by priority
        scores["priority_rank"] = scores["priority_score"].rank(
            ascending=False, method="min"
        ).astype(int)

        # Generate recommended action
        scores["recommended_action"] = scores.apply(
            self._generate_action, axis=1
        )

        return scores.sort_values("priority_rank")

    def _generate_action(self, row: pd.Series) -> str:
        """Generate action recommendation for a site."""
        actions = []

        if row["uplift_confident"]:
            actions.append(f"Upgrade hardware (expected +${row['expected_uplift']:,.0f}/mo)")

        if row["success_probability"] > 0.7:
            actions.append("High activation potential")
        elif row["success_probability"] < 0.3:
            actions.append("Low activation potential - investigate barriers")

        if row["lookalike_score"] > 0.8:
            actions.append("Strong lookalike to top performers")

        if not actions:
            return "Monitor - moderate potential"

        return "; ".join(actions)

    def get_top_recommendations(
        self,
        sites: pd.DataFrame,
        top_n: int = 100,
    ) -> List[SiteRecommendation]:
        """Get top N site recommendations.

        Args:
            sites: DataFrame with site features
            top_n: Number of recommendations to return

        Returns:
            List of SiteRecommendation objects
        """
        scores = self.prioritize(sites).head(top_n)

        recommendations = []
        for _, row in scores.iterrows():
            recommendations.append(
                SiteRecommendation(
                    site_id=row["site_id"],
                    priority_rank=row["priority_rank"],
                    priority_score=row["priority_score"],
                    lookalike_score=row["lookalike_score"],
                    success_probability=row["success_probability"],
                    expected_uplift=row["expected_uplift"],
                    uplift_confident=row["uplift_confident"],
                    recommended_action=row["recommended_action"],
                )
            )

        return recommendations

    def explain_recommendation(
        self,
        site: pd.Series,
        sites_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Explain why a site received its recommendation.

        Args:
            site: Single site Series
            sites_df: Full sites DataFrame for context

        Returns:
            Explanation dict with model-specific insights
        """
        site_df = site.to_frame().T

        # Get scores
        lookalike = self.similarity_model.score(site_df)[0]
        success_prob = self.classifier.predict_proba(site_df)[0]
        effect = self.causal_model.effect(site_df)[0]
        lower, upper = self.causal_model.effect_interval(site_df, alpha=0.05)

        # Get nearest high performers for similarity explanation
        nearest_performers, distances = self.similarity_model.get_nearest_performers(
            site, n=3
        )

        # Get classifier explanation
        classifier_explanation = self.classifier.explain_prediction(site_df, 0)

        # Get causal recommendation
        causal_recommendation = self.causal_model.generate_recommendation(site)

        return {
            "site_id": site.get("site_id", "unknown"),
            "scores": {
                "lookalike": lookalike,
                "success_probability": success_prob,
                "expected_uplift": effect,
                "uplift_ci": [lower[0], upper[0]],
            },
            "similarity_explanation": {
                "nearest_performers": nearest_performers["site_id"].tolist(),
                "distances": distances.tolist(),
            },
            "classifier_explanation": classifier_explanation,
            "causal_recommendation": causal_recommendation,
        }

    def save_recommendations_to_db(
        self,
        scores: pd.DataFrame,
        run_id: str,
        connection_string: str,
    ) -> int:
        """Save recommendations to database.

        Args:
            scores: DataFrame from prioritize()
            run_id: Model run ID for tracking
            connection_string: Database connection string

        Returns:
            Number of rows inserted
        """
        from sqlalchemy import create_engine

        engine = create_engine(connection_string)

        # Prepare data for insertion
        predictions = scores.copy()
        predictions["run_id"] = run_id

        # Insert into predictions table
        predictions.to_sql(
            "predictions",
            engine,
            schema="dooh",
            if_exists="append",
            index=False,
        )

        return len(predictions)

    def to_json(self, sites: pd.DataFrame, top_n: int = 100) -> str:
        """Export top recommendations as JSON.

        Args:
            sites: DataFrame with site features
            top_n: Number of recommendations

        Returns:
            JSON string of recommendations
        """
        import json

        recommendations = self.get_top_recommendations(sites, top_n)
        return json.dumps(
            [r.to_dict() for r in recommendations],
            indent=2,
        )
