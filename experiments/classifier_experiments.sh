#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# parame are:
# top_n workshops
# use abstract
# use n2v vector
# n2v type
# intent smoothing

# Run with abstract only
bash "$DIR"/train_venue_classifier.sh 25 true false

# Explore parameter space of intent smoothing with only n2v vector
bash "$DIR"/train_venue_classifier.sh 25 false true combined 0.5 &&
bash "$DIR"/train_venue_classifier.sh 25 false true combined 0.1 &&
bash "$DIR"/train_venue_classifier.sh 25 false true combined 0.01 &&

# Explore parameter space for abstract + n2v
bash "$DIR"/train_venue_classifier.sh 25 true true combined 0.5 &&
bash "$DIR"/train_venue_classifier.sh 25 true true combined 0.1 &&
bash "$DIR"/train_venue_classifier.sh 25 true true combined 0.01

# Explore parameter space for abstract + n2v
bash "$DIR"/train_venue_classifier.sh 25 true true all &&
bash "$DIR"/train_venue_classifier.sh 25 true true all &&
bash "$DIR"/train_venue_classifier.sh 25 true true all
