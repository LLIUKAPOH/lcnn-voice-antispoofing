import pandas as pd


class MetricTracker:
    """aggregate metrics from many batches"""

    def __init__(self, *keys, writer=None):
        """*keys are names of losses and writer is Comet"""
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        """reset all metrics after epoch end"""
        self._data.loc[:, :] = 0

    def update(self, key, value, n=1):
        """key - name of metric, Value - of metric on the batch, n - times to count value"""
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()
