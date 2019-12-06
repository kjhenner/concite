ROOT_DIR=/shared/1/projects/concite/
CUDA_DEVICE=2

#for MODEL in "model" \
#  "model_BERT" \
#  "model_BERT_n2v_all_prop" \
#  "model_BERT_n2v_all_uniform" \
#  "model_BERT_n2v_bkg_penalty" \
#  "model_BERT_n2v_combined_fixed" \
#  "model_BERT_n2v_combined_prop" \
#  "model_n2v_all_prop" \
#  "model_n2v_all_uniform" \
#  "model_n2v_bkg_penalty" \
#  "model_n2v_combined_fixed" \
for MODEL in "model_n2v_combined_prop"
do
  allennlp evaluate \
    "$ROOT_DIR"serialization/sequence_serialization/666/"$MODEL"/model.tar.gz \
    "$ROOT_DIR"data/acl_data/dev_acl_citation_sequences.txt \
    --include-package concite \
    --output-file output/sequence/"$MODEL".jsonl \
    --overrides '{"model": {"calculate_recall":true}}' \
    --cuda-device $CUDA_DEVICE
done
