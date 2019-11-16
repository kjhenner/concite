#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#bash "$DIR"/train_embedder.sh title true false &&

bash "$DIR"/train_embedder.sh title false true all &&
#bash "$DIR"/train_embedder.sh title true true all &&

bash "$DIR"/train_embedder.sh title false true combined
#bash "$DIR"/train_embedder.sh title true true combined
