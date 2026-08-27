"""POD-orientation metrics (experimental).

Both metrics are thin readers over the engine-side POD computation
(``src.orientation.compute_pod_data``), whose result the engine attaches to
``MetricContext.pod_orientation`` — mirroring how the pod / threat-clearance /
room-coverage compute families feed their metrics.

- ``POD_SECTOR_COVERAGE`` — fraction of the room area inside the union of the
  team's muzzle-direction sectors at the POD-establishment frame.
- ``POD_MUTUAL_FACING`` — 1 − (members whose sector contains at least one
  teammate) / team size at that frame.

Both return the repo's ``-1`` sentinel when POD establishment is uncertain
(no collective pause found, or too few members with reliable data at the
POD frame).
"""

from typing import Any, Dict, Optional

from .metric import AbstractMetric


class _PodOrientationBase(AbstractMetric):
    _metric_key = ""

    def __init__(self, config):
        super().__init__(config)
        self._data: Optional[Dict[str, Any]] = None

    def process(self, ctx) -> None:
        self._data = getattr(ctx, "pod_orientation", None)

    def getFinalScore(self) -> float:
        if not isinstance(self._data, dict):
            return -1
        score = (self._data.get("metrics") or {}).get(self._metric_key, -1)
        try:
            return float(score)
        except (TypeError, ValueError):
            return -1


class PodSectorCoverage_Metric(_PodOrientationBase):
    _metric_key = "POD_SECTOR_COVERAGE"

    def __init__(self, config):
        super().__init__(config)
        self.metricName = "POD_SECTOR_COVERAGE"


class PodMutualFacing_Metric(_PodOrientationBase):
    _metric_key = "POD_MUTUAL_FACING"

    def __init__(self, config):
        super().__init__(config)
        self.metricName = "POD_MUTUAL_FACING"
