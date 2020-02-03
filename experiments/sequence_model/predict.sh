ROOT_DIR=/shared/1/projects/concite/
CUDA_DEVICE=$1

allennlp predict \
  "$ROOT_DIR"sequence_serialization_100/citations_"$MODEL"/model.tar.gz \
  /home/khenner/src/context_net/experiments/sequence_model/case_study \
  --include-package concite \
  --predictor sequence \
  --output-file output/sequence/predictions/"$MODEL".jsonl \
  --overrides '{"model": {"calculate_recall": true}}' \
  --use-dataset-reader \
  --cuda-device $CUDA_DEVICE
