from typing import Dict
from copy import deepcopy

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.predictors.predictor import Predictor

@Predictor.register("sequence_demo")
class SequenceDemoPredictor(Predictor):

    def format_output(self, paper_id):
        if paper_id == "<s>":
            return {"id": "S"}
        title = self._dataset_reader.data_lookup[paper_id]['abstract']
        if paper_id[:3] == "Ext":
            return {"id": paper_id, "title": title}
        else:
            url = "https://www.aclweb.org/anthology/{}/".format(paper_id)
            return {"id": paper_id, "title": title, "url": url}

    def token_to_title(self, token):
        if token.text == "<s>":
            return "none"
        return self._dataset_reader.data_lookup[token.text]['abstract']

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        new_instance = deepcopy(instance)
        outputs = self._model.forward_on_instance(new_instance)
        print(outputs)
        return {
                "titles": [self.token_to_title(token) for token in instance['paper_ids'].tokens],
                "top_predictions": [self.format_output(paper_id)
                    for paper_id in outputs['top_k'][-1][:10]]
                }

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(json_dict['sequence'])
