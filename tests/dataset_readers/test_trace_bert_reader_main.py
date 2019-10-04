from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

import sys

from pathlib import Path

sys.path.append(str(Path('.').absolute()))

from concite.dataset_readers.aclarc_trace_data_reader import AclarcTraceDatasetReader

class TestTraceBertDatasetReader(AllenNlpTestCase):

    def test_read_from_acl_trace_file(self):
        reader = AclarcTraceDatasetReader(abstract_lookup_path='tests/fixtures/acl_n2v.jsonl')
        instances = ensure_list(reader.read('tests/fixtures/train_arc_sessions.with-ids.txt'))
        print(len(instances))
        assert isinstance(instances, list)
