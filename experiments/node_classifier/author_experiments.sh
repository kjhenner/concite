#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

bash "$DIR"/train_acl_classifier.sh \
  8 last_author 100 true false &&

bash "$DIR"/train_acl_classifier.sh \
  8 last_author 100 false true all 20 0.3 0.7 &&

bash "$DIR"/train_acl_classifier.sh \
  8 last_author 100 false true combined 20 0.3 0.7 0.5 &&

bash "$DIR"/train_acl_classifier.sh \
  8 last_author 100 true true all 20 0.3 0.7 &&

bash "$DIR"/train_acl_classifier.sh \
  8 last_author 100 true true combined 20 0.3 0.7 0.5
