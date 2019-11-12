allennlp evaluate \
  /shared-1/projects/concite/sequence_serialization_100/citations_n2v_all_20_384_0.3_0.7/model.tar.gz \
  /shared-1/projects/concite/data/acl_data/dev_acl_citation_sequences.txt \
  --include-package concite \
  --output-file pred.out \
  --cuda-device 6
