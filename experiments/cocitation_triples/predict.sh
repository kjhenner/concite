CUDA_DEVICE=$1
MODEL=model_n2v_all_uniform

allennlp predict \
  /shared/0/projects/concite/serialization/cocitation_serialization/666/"$MODEL"/model.tar.gz \
  /shared/1/projects/concite/data/acl_data/sentence_co-occurrence.txt \
  --include-package concite \
  --predictor cocitation \
  --output-file output/cocitation/"$MODEL".jsonl \
  --use-dataset-reader \
  --cuda-device $CUDA_DEVICE
