#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export CUDA_DEVICE=$1

for SEED in 666 #669 672 675 678
do
  export SEED=$SEED
  export PYTORCH_SEED=$SEED
  export NUMPY_SEED=$SEED

  bash "$DIR"/train_acl_glove_classifier.sh \
    10 combined_workshop 100 false &&

  bash "$DIR"/train_acl_glove_classifier.sh \
    10 combined_workshop 100 true all

#  bash "$DIR"/train_acl_classifier.sh \
#    10 combined_workshop 100 true combined &&
#
#  bash "$DIR"/train_acl_classifier.sh \
#    8 last_author 100 false &&
#
#  bash "$DIR"/train_acl_classifier.sh \
#    8 last_author 100 true all &&
#
#  bash "$DIR"/train_acl_classifier.sh \
#    8 last_author 100 true combined &&

done
