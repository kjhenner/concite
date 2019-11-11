#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

bash "$DIR"/train_acl_edge_classifier.sh 3 intent_label abstract 100 true false #82 83
