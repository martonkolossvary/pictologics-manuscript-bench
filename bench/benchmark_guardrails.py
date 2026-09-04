from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class GuardrailPolicy:
    enabled: bool = False
    baseline_adapter: str = "pictologics"
    skip_ratio: float = 1000.0
    minimum_slow_observations: int = 1
    truncate_on_timeout: bool = True

    def __post_init__(self) -> None:
        if not self.baseline_adapter.strip():
            raise ValueError("guardrail baseline adapter cannot be empty")
        if self.skip_ratio <= 0:
            raise ValueError("guardrail skip ratio must be positive")
        if self.minimum_slow_observations < 1:
            raise ValueError("guardrail minimum observations must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "baseline_adapter": self.baseline_adapter,
            "skip_ratio": self.skip_ratio,
            "minimum_slow_observations": self.minimum_slow_observations,
            "speed_truncation": {
                "enabled": self.enabled,
                "strictly_larger_only": True,
                "scope_dimensions": ["adapter", "workload", "guardrail_group"],
            },
            "timeout_cutoff": {
                "enabled": self.truncate_on_timeout,
                "strictly_larger_only": True,
                "complexity_metric": "image_voxels",
                "scope_dimensions": ["adapter", "workload", "guardrail_group"],
                "applies_to_baseline": True,
            },
        }

    def scope_key(
        self,
        adapter: str,
        workload_key: str,
        guardrail_group: str,
    ) -> str:
        if not guardrail_group.strip():
            raise ValueError("guardrail group cannot be empty")
        return f"{adapter}\x1f{workload_key}\x1f{guardrail_group}"

    def should_compare(self, adapter: str) -> bool:
        return self.enabled and adapter != self.baseline_adapter

    @staticmethod
    def timeout_scope_key(
        adapter: str,
        workload_key: str,
        guardrail_group: str,
    ) -> str:
        if not adapter.strip() or not workload_key.strip() or not guardrail_group.strip():
            raise ValueError("timeout cutoff scope dimensions cannot be empty")
        return f"{adapter}\x1f{workload_key}\x1f{guardrail_group}"

    def ratio(
        self, duration: Optional[float], baseline: Optional[float]
    ) -> Optional[float]:
        if duration is None or baseline is None or duration <= 0 or baseline <= 0:
            return None
        return float(duration) / float(baseline)
