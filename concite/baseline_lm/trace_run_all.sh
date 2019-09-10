#!/bin/sh

./ngram_count.sh ../../data/train_arc-sessions.with-ids.txt ../../data/train_arc-sessions.with-ids.ngram_count &&
./build_lm.sh ../../data/train_arc-sessions.with-ids.ngram_count ../../data/train_arc-sessions.with-ids.lm &&
./ppl.sh ../../data/train_arc-sessions.with-ids.lm 0.05 0.95 0 ../../data/dev_arc-sessions.with-ids.txt trace_ppl_0.05_0.95_0
./ppl.sh ../../data/train_arc-sessions.with-ids.lm 0.05 0.2 0.75 ../../data/dev_arc-sessions.with-ids.txt trace_ppl_0.05_0.2_0.75
./ppl.sh ../../data/train_arc-sessions.with-ids.lm 1 0 0 ../../data/dev_arc-sessions.with-ids.txt trace_ppl_unigram
