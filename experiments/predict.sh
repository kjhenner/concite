allennlp predict \
  /shared-1/projects/concite/serialization_100/10_combined_workshop_abstract/model.tar.gz \
  /shared-1/projects/concite/data/acl_data/dev_10_combined_workshop_acl_data.jsonl \
  --include-package concite \
  --predictor acl_classifier \
  --output-file workshops.out \
  --cuda-device 3 \
  --use-dataset-reader \
