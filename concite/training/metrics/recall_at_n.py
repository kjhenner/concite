from typing import Dict
from overrides import overrides
from collections import defaultdict
import torch

from allennlp.training.metrics.metric import Metric

@Metric.register("recall_at_n")
class RecallAtN(Metric):

    def __init__(self, n=5) -> None:
        self._tp = 0
        self._count = 0
        self._n = n

    @overrides
    def reset(self):
        self._tp = 0
        self._count = 0

    @overrides
    def __call__(self, targets, top_indices):
        self._count += 1
        for target, predictions in zip(targets, top_indices[:self._n]):
            if target in predictions:
                self._tp += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[int, float]:
        if self._count == 0:
            recall = 0.0
        else:
            recall = float(self._tp) / self._count
        if reset:
            self.reset()
        return recall
