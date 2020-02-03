#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export CUDA_DEVICE=$1

for SEED in 666 669 672 675 678
do
  export SEED=$SEED
  export PYTORCH_SEED=$SEED
  export NUMPY_SEED=$SEED

  bash "$DIR"/train_acl_classifier.sh \
    10 combined_workshop 100 true false &&

  bash "$DIR"/train_acl_classifier.sh \
    10 combined_workshop 100 true true all_prop &&

  bash "$DIR"/train_acl_classifier.sh \
    10 combined_workshop 100 true true all_uniform &&

  bash "$DIR"/train_acl_classifier.sh \
    10 combined_workshop 100 true true bkg_penalty &&

  bash "$DIR"/train_acl_classifier.sh \
    10 combined_workshop 100 true true combined_fixed &&

  bash "$DIR"/train_acl_classifier.sh \
    10 combined_workshop 100 true true combined_prop

done
