from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

import sys

from pathlib import Path

sys.path.append(str(Path('.').absolute()))

from concite.dataset_readers.acl_graph_reader import AclGraphReader

class TestDatasetReader(AllenNlpTestCase):

    def test_acl_graph_reader(self):
        reader = AclGraphReader(
                label_field='workshop',
                text_field='abstract'
        )
        instances = ensure_list(reader.read('tests/fixtures/test_node_data.jsonl'))
        print(instances)
        print(len(instances))
        assert isinstance(instances, list)
