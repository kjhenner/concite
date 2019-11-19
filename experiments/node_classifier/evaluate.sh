ROOT_DIR=/shared-1/projects/concite/
CUDA_DEVICE=$1

for MODEL in "abstract" \
  "abstract_n2v_all_20_384_0.3_0.7" \
  "abstract_n2v_combined_20_384_0.3_0.7_0.5" \
  "n2v_all_20_384_0.3_0.7" \
  "n2v_combined_20_384_0.3_0.7_0.5"
do
  allennlp evaluate \
    "$ROOT_DIR"serialization_100/8_last_author_"$MODEL"/model.tar.gz \
    "$ROOT_DIR"data/acl_data/test_8_last_author_acl_data.jsonl \
    --include-package concite \
    --output-file last_author_"$MODEL".json \
    --cuda-device $CUDA_DEVICE
done
