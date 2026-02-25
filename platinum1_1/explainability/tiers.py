"""
Executive Tier Classification for Site Scoring.

Translates calibrated probabilities into business-friendly tiers with
historical accuracy context.

Tier System:
- Tier 1 (Recommended): >85% confidence, ~88% historical accuracy
- Tier 2 (Promising): 65-85% confidence, ~76% historical accuracy
- Tier 3 (Review Required): 50-65% confidence, ~62% historical accuracy
- Tier 4 (Not Recommended): <50% confidence
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np

TIER_THRESHOLDS = [0.85, 0.65, 0.50]

TIER_LABELS = {1: "Recommended", 2: "Promising", 3: "Review Required", 4: "Not Recommended"}
TIER_ACTIONS = {1: "Proceed to contract", 2: "Site visit required", 3: "Detailed assessment needed", 4: "Do not pursue"}
TIER_COLORS = {1: "#22c55e", 2: "#eab308", 3: "#f97316", 4: "#ef4444"}


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
    """Maps calibrated probabilities to executive-friendly tiers."""

    def __init__(
        self,
        thresholds: List[float] = None,
        historical_accuracy: Dict[int, float] = None,
    ):
        self.thresholds = thresholds or TIER_THRESHOLDS.copy()

        if not all(self.thresholds[i] > self.thresholds[i + 1]
                   for i in range(len(self.thresholds) - 1)):
            raise ValueError("Thresholds must be in descending order")

        self.historical_accuracy = historical_accuracy or {
            1: 0.88, 2: 0.76, 3: 0.62, 4: None,
        }
        self._tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        self._tier_outcomes = {1: [], 2: [], 3: [], 4: []}

    def classify(self, calibrated_prob: float) -> TierResult:
        """Classify a single calibrated probability into a tier."""
        if calibrated_prob >= self.thresholds[0]:
            tier = 1
        elif calibrated_prob >= self.thresholds[1]:
            tier = 2
        elif calibrated_prob >= self.thresholds[2]:
            tier = 3
        else:
            tier = 4

        confidence_statement = self._generate_confidence_statement(calibrated_prob, tier)
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
        """Record actual outcome for a site to update historical accuracy."""
        self._tier_outcomes[tier].append(1 if succeeded else 0)

    def update_historical_accuracy(self) -> Dict[int, float]:
        """Recalculate historical accuracy from recorded outcomes."""
        for tier in [1, 2, 3, 4]:
            outcomes = self._tier_outcomes[tier]
            if len(outcomes) >= 10:
                self.historical_accuracy[tier] = np.mean(outcomes)
        return self.historical_accuracy.copy()

    def get_tier_distribution(self, calibrated_probs: np.ndarray) -> Dict[int, Tuple[int, float]]:
        """Get distribution of sites across tiers."""
        tiers = np.digitize(calibrated_probs, bins=[0] + self.thresholds[::-1] + [1.01])
        tiers = 5 - tiers
        n_total = len(calibrated_probs)
        return {
            tier: (int((tiers == tier).sum()), (tiers == tier).sum() / n_total if n_total > 0 else 0)
            for tier in [1, 2, 3, 4]
        }

    def to_dict(self) -> dict:
        return {
            "thresholds": self.thresholds,
            "historical_accuracy": self.historical_accuracy,
            "tier_counts": self._tier_counts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TierClassifier':
        instance = cls(
            thresholds=data.get("thresholds"),
            historical_accuracy=data.get("historical_accuracy"),
        )
        instance._tier_counts = data.get("tier_counts", {1: 0, 2: 0, 3: 0, 4: 0})
        return instance
