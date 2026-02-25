class AbstractMetric:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.metricName = "Metric"
    def process(self, ctx):
        """
        Process the unified MetricContext for this metric.
        Subclasses must override this to extract and prepare needed data.
        """
        raise NotImplementedError("Subclasses must implement process")
    def updateFrame(self, map_pos):
        pass
    def getFinalScore(self) -> float:
        pass
    @staticmethod
    def expertCompare(expert_track, trainee_track, mapView, outputFolder):
        pass