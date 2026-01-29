"""
Executive Tier Classification for Site Scoring.

Translates calibrated probabilities into business-friendly tiers with
historical accuracy context. Designed for non-technical stakeholders
who need actionable recommendations, not probability scores.

Tier System:
- Tier 1 (Recommended): >85% confidence, ~88% historical accuracy
- Tier 2 (Promising): 65-85% confidence, ~76% historical accuracy
- Tier 3 (Review Required): 50-65% confidence, ~62% historical accuracy
- Tier 4 (Not Recommended): <50% confidence

Reference: Based on risk stratification practices in financial services.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np

# Default tier configuration
TIER_THRESHOLDS = [0.85, 0.65, 0.50]  # Boundaries between tiers

TIER_LABELS = {
    1: "Recommended",
    2: "Promising",
    3: "Review Required",
    4: "Not Recommended",
}

TIER_ACTIONS = {
    1: "Proceed to contract",
    2: "Site visit required",
    3: "Detailed assessment needed",
    4: "Do not pursue",
}

TIER_COLORS = {
    1: "#22c55e",  # Green
    2: "#eab308",  # Yellow
    3: "#f97316",  # Orange
    4: "#ef4444",  # Red
}


@dataclass
class TierResult:
    """Result of tier classification for a single site."""
    tier: int
    label: str
    action: str
    calibrated_probability: float
    confidence_statement: str
    historical_accuracy: Optional[float] = None
    color: str = ""

    def __post_init__(self):
        self.color = TIER_COLORS.get(self.tier, "#6b7280")

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tier": self.tier,
            "label": self.label,
            "action": self.action,
            "calibrated_probability": self.calibrated_probability,
            "confidence_statement": self.confidence_statement,
            "historical_accuracy": self.historical_accuracy,
            "color": self.color,
        }


class TierClassifier:
    """
    Maps calibrated probabilities to executive-friendly tiers.

    Key Design Principle:
    ---------------------
    Executives don't need to understand probability scores. They need
    to know: "Should we pursue this site?" and "How confident are we?"

    The tier system provides:
    1. Clear categorical recommendation (Recommended, Promising, etc.)
    2. Confidence in plain language ("8 out of 10 similar sites succeeded")
    3. Recommended action (Proceed, Visit, Review, Skip)
    4. Historical accuracy for accountability

    Args:
        thresholds: List of 3 probability thresholds [tier1, tier2, tier3]
                   Default: [0.85, 0.65, 0.50]

    Example:
        >>> classifier = TierClassifier()
        >>> result = classifier.classify(0.78)
        >>> print(f"{result.label}: {result.confidence_statement}")
        Promising: 7-8 out of 10 similar sites succeeded
    """

    def __init__(
        self,
        thresholds: List[float] = None,
        historical_accuracy: Dict[int, float] = None,
    ):
        self.thresholds = thresholds or TIER_THRESHOLDS.copy()

        # Validate thresholds are descending
        if not all(self.thresholds[i] > self.thresholds[i + 1]
                   for i in range(len(self.thresholds) - 1)):
            raise ValueError("Thresholds must be in descending order")

        # Default historical accuracy (should be updated with actual data)
        self.historical_accuracy = historical_accuracy or {
            1: 0.88,  # 88% of Tier 1 sites succeeded
            2: 0.76,  # 76% of Tier 2 sites succeeded
            3: 0.62,  # 62% of Tier 3 sites succeeded
            4: None,  # Not tracked for Tier 4
        }

        # Statistics for tracking
        self._tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        self._tier_outcomes = {1: [], 2: [], 3: [], 4: []}

    def classify(self, calibrated_prob: float) -> TierResult:
        """
        Classify a single calibrated probability into a tier.

        Args:
            calibrated_prob: Calibrated probability in [0, 1]

        Returns:
            TierResult with tier, label, action, and confidence statement
        """
        # Determine tier
        if calibrated_prob >= self.thresholds[0]:
            tier = 1
        elif calibrated_prob >= self.thresholds[1]:
            tier = 2
        elif calibrated_prob >= self.thresholds[2]:
            tier = 3
        else:
            tier = 4

        # Generate confidence statement
        confidence_statement = self._generate_confidence_statement(
            calibrated_prob, tier
        )

        self._tier_counts[tier] += 1

        return TierResult(
            tier=tier,
            label=TIER_LABELS[tier],
            action=TIER_ACTIONS[tier],
            calibrated_probability=calibrated_prob,
            confidence_statement=confidence_statement,
            historical_accuracy=self.historical_accuracy.get(tier),
        )

    def classify_batch(self, calibrated_probs: np.ndarray) -> List[TierResult]:
        """Classify multiple probabilities."""
        return [self.classify(p) for p in calibrated_probs]

    def _generate_confidence_statement(self, prob: float, tier: int) -> str:
        """
        Generate executive-friendly confidence statement.

        Converts probability to "X out of 10" format which is more
        intuitive than percentages for non-technical audiences.
        """
        # Convert to X out of 10
        out_of_10 = int(round(prob * 10))

        if tier == 1:
            return f"{out_of_10} out of 10 similar sites succeeded at this confidence level"
        elif tier == 2:
            return f"{out_of_10} out of 10 similar sites succeeded"
        elif tier == 3:
            return f"Only {out_of_10} out of 10 similar sites succeeded - review recommended"
        else:
            return f"Less than {out_of_10} out of 10 similar sites succeed at this level"

    def record_outcome(self, tier: int, succeeded: bool) -> None:
        """
        Record actual outcome for a site to update historical accuracy.

        Call this after a site's performance is known to improve
        future accuracy estimates.

        Args:
            tier: The tier the site was classified into
            succeeded: Whether the site met success criteria
        """
        self._tier_outcomes[tier].append(1 if succeeded else 0)

    def update_historical_accuracy(self) -> Dict[int, float]:
        """
        Recalculate historical accuracy from recorded outcomes.

        Should be called periodically (e.g., quarterly) to keep
        accuracy estimates current.

        Returns:
            Updated accuracy dict
        """
        for tier in [1, 2, 3, 4]:
            outcomes = self._tier_outcomes[tier]
            if len(outcomes) >= 10:  # Minimum sample size
                self.historical_accuracy[tier] = np.mean(outcomes)

        return self.historical_accuracy.copy()

    def get_tier_distribution(
        self,
        calibrated_probs: np.ndarray
    ) -> Dict[int, Tuple[int, float]]:
        """
        Get distribution of sites across tiers.

        Returns:
            Dict mapping tier -> (count, percentage)
        """
        tiers = np.digitize(
            calibrated_probs,
            bins=[0] + self.thresholds[::-1] + [1.01],
        )
        # Flip because digitize uses ascending order
        tiers = 5 - tiers

        n_total = len(calibrated_probs)
        distribution = {}

        for tier in [1, 2, 3, 4]:
            count = (tiers == tier).sum()
            distribution[tier] = (int(count), count / n_total if n_total > 0 else 0)

        return distribution

    def get_summary_table(self) -> str:
        """
        Generate ASCII summary table of tier definitions.

        Useful for documentation and executive presentations.
        """
        lines = [
            "┌─────────┬───────────────────┬─────────────────────┬────────────────────────┐",
            "│  Tier   │    Score Range    │   Historical Acc.   │   Recommended Action   │",
            "├─────────┼───────────────────┼─────────────────────┼────────────────────────┤",
        ]

        ranges = [
            (1, f">{self.thresholds[0]:.0%}"),
            (2, f"{self.thresholds[1]:.0%}-{self.thresholds[0]:.0%}"),
            (3, f"{self.thresholds[2]:.0%}-{self.thresholds[1]:.0%}"),
            (4, f"<{self.thresholds[2]:.0%}"),
        ]

        for tier, range_str in ranges:
            label = TIER_LABELS[tier]
            acc = self.historical_accuracy.get(tier)
            acc_str = f"{acc:.0%}" if acc else "N/A"
            action = TIER_ACTIONS[tier]

            lines.append(
                f"│ {tier} {label:<6} │ {range_str:^17} │ {acc_str:^19} │ {action:<22} │"
            )

        lines.append(
            "└─────────┴───────────────────┴─────────────────────┴────────────────────────┘"
        )

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export configuration for serialization."""
        return {
            "thresholds": self.thresholds,
            "historical_accuracy": self.historical_accuracy,
            "tier_counts": self._tier_counts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TierClassifier':
        """Create from serialized dict."""
        instance = cls(
            thresholds=data.get("thresholds"),
            historical_accuracy=data.get("historical_accuracy"),
        )
        instance._tier_counts = data.get("tier_counts", {1: 0, 2: 0, 3: 0, 4: 0})
        return instance
