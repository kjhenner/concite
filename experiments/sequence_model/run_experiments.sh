#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export CUDA_DEVICE=$1

for SEED in 666 669 672 675 678
do
  export SEED=$SEED
  export PYTORCH_SEED=$SEED
  export NUMPY_SEED=$SEED

  bash "$DIR"/train_acl_sequence_model.sh \
    title 100 false false &&

  bash "$DIR"/train_acl_sequence_model.sh \
    title 100 true false &&

  bash "$DIR"/train_acl_sequence_model.sh \
    title 100 false true all&&

  bash "$DIR"/train_acl_sequence_model.sh \
    title 100 false true combined &&

  bash "$DIR"/train_acl_sequence_model.sh \
    title 100 true true all &&

  bash "$DIR"/train_acl_sequence_model.sh \
    title 100 true true combined

done
