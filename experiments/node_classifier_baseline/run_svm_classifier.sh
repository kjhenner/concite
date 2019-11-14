#!/bin/bash    

BASE_DIR=/shared-1/projects/concite/data/

LABEL_FIELD=combined_workshop
SERIALIZATION_DIR="$BASE_DIR"svm_baseline/"$LABEL_FIELD"

mkdir -p $SERIALIZATION_DIR

python /home/khenner/src/context_net/scripts/baselines/svm_classifier.py \
  "$BASE_DIR"acl_data/train_10_"$LABEL_FIELD"_acl_data.jsonl \
  "$BASE_DIR"acl_data/test_10_"$LABEL_FIELD"_acl_data.jsonl \
  "$BASE_DIR"acl_data/tfidf_acl_data.jsonl \
  $LABEL_FIELD \
  $SERIALIZATION_DIR > "$SERIALIZATION_DIR"/summary.txt &&

LABEL_FIELD=last_author
SERIALIZATION_DIR="$BASE_DIR"svm_baseline/"$LABEL_FIELD"

mkdir -p $SERIALIZATION_DIR

python /home/khenner/src/context_net/scripts/baselines/svm_classifier.py \
  "$BASE_DIR"acl_data/train_8_"$LABEL_FIELD"_acl_data.jsonl \
  "$BASE_DIR"acl_data/test_8_"$LABEL_FIELD"_acl_data.jsonl \
  "$BASE_DIR"acl_data/tfidf_acl_data.jsonl \
  $LABEL_FIELD \
  $SERIALIZATION_DIR > "$SERIALIZATION_DIR"/summary.txt
