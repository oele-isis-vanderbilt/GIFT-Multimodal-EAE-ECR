# Several metrics build comparison charts via matplotlib.pyplot. When the
# metrics package is loaded inside the analysis-viewer sidecar (FastAPI in a
# headless process) the default macOS backend tries to create a GUI canvas
# and crashes ("NSWindow can only be created on the main thread"). Force the
# non-interactive Agg backend before any submodule imports pyplot.
import matplotlib as _mpl
_mpl.use("Agg")

from .context import MetricContext
from .entrance_hesitation import EntranceHesitation_Metric
from .entrance_vectors import EntranceVectors_Metric
from .move_along_wall import MoveAlongWall_Metric
from .pod import POD_Metric
from .total_entry_time import TotalEntryTime_Metric
from .metric import AbstractMetric

from .threat_clearance import ThreatClearance_Metric
from .teammate_coverage import TeammateCoverage_Metric
from .threat_coverage import ThreatCoverage_Metric
from .room_coverage import RoomCoverage_Metric
from .room_coverage_time import TotalRoomCoverageTime_Metric
from .capture_pod import IdentifyAndCapturePods_Metric
from .capture_pod_time import CapturePodTime_Metric
from .pod_orientation import PodMutualFacing_Metric, PodSectorCoverage_Metric