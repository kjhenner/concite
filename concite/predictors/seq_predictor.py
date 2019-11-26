from typing import Dict
from copy import deepcopy

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.predictors.predictor import Predictor

@Predictor.register("sequence")
class SequencePredictor(Predictor):

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        new_instance = deepcopy(instance)
        outputs = self._model.forward_on_instance(new_instance)
        output_dict = {
                'top_k_titles': [self._dataset_reader.data_lookup[paper_id]
                    for positions in outputs['top_k']
                    for paper_id in positions],
                'top_k': [paper_id if paper_id != "<s>" else "S"
                    for positions in outputs['top_k']
                    for paper_id in positions],
                }
        return sanitize(output_dict)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(json_dict['sequence'])
