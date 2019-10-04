from overrides import overrides
import torch

from allennlp.training.metrics.average import Average
from allennlp.training.metrics.metric import Metric

@Metric.register("perplexity")
class Perplexity(Average):

    @overrides
    def get_metric(self, reset: bool = False) -> float:
        average_loss = super().get_metric(reset)
        if average_loss == 0:
            return 0.0

        return float(torch.exp(average_loss))
