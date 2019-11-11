#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#82 83
bash "$DIR"/train_acl_classifier.sh 10 combined_workshop 100 true false &&

#72 75
bash "$DIR"/train_acl_classifier.sh 10 combined_workshop 100 false true all 20 0.3 0.7 &&
#82 84
bash "$DIR"/train_acl_classifier.sh 10 combined_workshop 100 true true all 20 0.3 0.7 &&

#72 73
bash "$DIR"/train_acl_classifier.sh 10 combined_workshop 100 false true combined 20 0.3 0.7 0.5 &&
#83 84
bash "$DIR"/train_acl_classifier.sh 10 combined_workshop 100 true true combined 20 0.3 0.7 0.5
