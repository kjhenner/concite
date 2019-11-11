#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# parame are:
# top_n authors
# use abstract
# use n2v vector
# n2v type
# intent smoothing

# Run with abstract only
bash "$DIR"/train_author_classifier.sh 100 10 true false &&
bash "$DIR"/train_author_classifier.sh 100 10 false true all 20 0.3 0.7 &&
bash "$DIR"/train_author_classifier.sh 100 10 false true combined 20 0.3 0.7 0.5 &&
bash "$DIR"/train_author_classifier.sh 100 10 true true all 20 0.3 0.7 &&
bash "$DIR"/train_author_classifier.sh 100 10 true true combined 20 0.3 0.7 0.5
