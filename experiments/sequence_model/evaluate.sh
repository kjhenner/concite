ROOT_DIR=/shared-1/projects/concite/
CUDA_DEVICE=6

for MODEL in n2v_all_20_384_0.3_0.7 n2v_combined_20_384_0.3_0.7_0.5
do
  allennlp evaluate \
    "$ROOT_DIR"sequence_serialization_100/citations_"$MODEL"/model.tar.gz \
    "$ROOT_DIR"data/acl_data/dev_acl_citation_sequences.txt \
    --include-package concite \
    --output-file "$MODEL".out \
    --cuda-device $CUDA_DEVICE

  allennlp evaluate \
    "$ROOT_DIR"sequence_serialization_100/citations_"$MODEL"/model.tar.gz \
    "$ROOT_DIR"data/acl_data/dev_acl_citation_sequences.txt \
    --include-package concite \
    --output-file "$MODEL".out \
    --cuda-device $CUDA_DEVICE
done
