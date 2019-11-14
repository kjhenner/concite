allennlp predict \
  /shared-1/projects/concite/serialization_100/10_combined_workshop_abstract/model.tar.gz \
  /shared-1/projects/concite/data/acl_data/dev_10_combined_workshop_acl_data.jsonl \
  --include-package concite \
  --predictor acl_classifier \
  --output-file workshops.out \
  --cuda-device 3 \
  --use-dataset-reader \
ROOT_DIR=/shared-1/projects/concite/
CUDA_DEVICE=2

for MODEL in "abstract" \
  "abstract_n2v_all_20_384_0.3_0.7" \
  "abstract_n2v_combined_20_384_0.3_0.7_0.5" \
  "n2v_all_20_384_0.3_0.7" \
  "n2v_combined_20_384_0.3_0.7_0.5"
do
  allennlp predict \
    "$ROOT_DIR"serialization_100/10_combined_workshop_"$MODEL"/model.tar.gz \
    "$ROOT_DIR"data/acl_data/test_10_combined_workshop_acl_data.jsonl \
    --include-package concite \
    --predictor acl_classifier \
    --output-file output/workshop/predictions/"$MODEL".jsonl \
    --use-dataset-reader \
    --cuda-device $CUDA_DEVICE
done
