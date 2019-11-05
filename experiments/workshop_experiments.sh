#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# parame are:
# top_n workshops
# use abstract
# use n2v vector
# n2v type
# intent smoothing

# Run with abstract only
#bash "$DIR"/train_venue_classifier.sh 100 10 true false && # 70 73

#bash "$DIR"/train_venue_classifier.sh 100 10 false true all 20 0.7 0.7 &&
#bash "$DIR"/train_venue_classifier.sh 100 10 false true all 20 0.7 0.5 && #62 67
#bash "$DIR"/train_venue_classifier.sh 100 10 false true all 20 0.7 0.3 &&
#
#bash "$DIR"/train_venue_classifier.sh 100 10 false true all 20 0.5 0.7 && #58 65
#bash "$DIR"/train_venue_classifier.sh 100 10 false true all 20 0.5 0.5 && #57 64
#bash "$DIR"/train_venue_classifier.sh 100 10 false true all 20 0.5 0.3 && #58 65

#bash "$DIR"/train_venue_classifier.sh 100 10 false true all 20 0.3 0.7 && #63 66
#bash "$DIR"/train_venue_classifier.sh 100 10 false true all 20 0.3 0.5 && #60 66 
#bash "$DIR"/train_venue_classifier.sh 100 10 false true all 20 0.3 0.3 && #59 65

#bash "$DIR"/train_venue_classifier.sh 100 10 false true combined 20 0.7 0.7 0.5 && # 59 66
#bash "$DIR"/train_venue_classifier.sh 100 10 false true combined 20 0.7 0.5 0.5 && #59 66
#bash "$DIR"/train_venue_classifier.sh 100 10 false true combined 20 0.7 0.3 0.5 && #58 67

#bash "$DIR"/train_venue_classifier.sh 100 10 false true combined 20 0.5 0.7 0.5 && #59 66
#bash "$DIR"/train_venue_classifier.sh 100 10 false true combined 20 0.5 0.5 0.5 && #68 70
#bash "$DIR"/train_venue_classifier.sh 100 10 false true combined 20 0.5 0.3 0.5 && #55 63

#bash "$DIR"/train_venue_classifier.sh 100 10 false true combined 20 0.3 0.7 0.5 && #69 73
#bash "$DIR"/train_venue_classifier.sh 100 10 false true combined 20 0.3 0.5 0.5 && #57 65
#bash "$DIR"/train_venue_classifier.sh 100 10 false true combined 20 0.3 0.3 0.5 && #55 63

#bash "$DIR"/train_venue_classifier.sh 100 10 true true combined 20 0.3 0.7 0.5 && # 72 75
#bash "$DIR"/train_venue_classifier.sh 100 10 true true all 20 0.3 0.7 && # 72 73

#bash "$DIR"/train_venue_classifier.sh 100 16 true false && #63 64
#bash "$DIR"/train_venue_classifier.sh 100 16 true true combined 20 0.3 0.7 0.5 && #65 67
#bash "$DIR"/train_venue_classifier.sh 100 16 true true all 20 0.3 0.7 #62 64

#bash "$DIR"/train_venue_classifier.sh 100 10 false true all 15 0.3 0.7 && #52 61
#bash "$DIR"/train_venue_classifier.sh 100 10 false true all 20 0.3 0.7 && #49 55
#bash "$DIR"/train_venue_classifier.sh 100 10 false true all 25 0.3 0.7 && #64 68

#bash "$DIR"/train_venue_classifier.sh 100 10 false true combined 20 0.3 0.7 0.3 && # 60 67
#bash "$DIR"/train_venue_classifier.sh 100 10 false true combined 20 0.3 0.7 0.4 && # 65 68

#bash "$DIR"/train_venue_classifier.sh 100 10 false true combined 15 0.3 0.7 0.5 && # 63 68
#bash "$DIR"/train_venue_classifier.sh 100 10 false true combined 25 0.3 0.7 0.5 # 52 59

# Explore parameter space for abstract + n2v
#bash "$DIR"/train_venue_classifier.sh 100 10 true true combined 0.5
#bash "$DIR"/train_venue_classifier.sh 100 24 true true combined 0.1 &&
