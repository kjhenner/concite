#!/bin/sh

./ngram_count.sh ../../data/train_arc-sessions.with-ids.txt ../../data/train_arc-sessions.with-ids.ngram_count &&
./build_lm.sh ../../data/train_arc-sessions.with-ids.ngram_count ../../data/train_arc-sessions.with-ids.lm &&
./ppl.sh ../../data/arc-sessions.with-ids.lm 0.05 0.15 0.8 ../../data/test_arc-sessions.with-ids.txt ppl_0.05_0.15_0.8
./ppl.sh ../../data/arc-sessions.with-ids.lm 0.1 0.1 0.8 ../../data/test_arc-sessions.with-ids.txt ppl_0.1_0.1_0.8
./ppl.sh ../../data/arc-sessions.with-ids.lm 0.2 0.3 0.4 ../../data/test_arc-sessions.with-ids.txt ppl_0.2_0.3_0.5
./ppl.sh ../../data/arc-sessions.with-ids.lm 0.2 0.5 0.3 ../../data/test_arc-sessions.with-ids.txt ppl_0.2_0.5_0.3
./ppl.sh ../../data/arc-sessions.with-ids.lm 0.2 0.7 0.1 ../../data/test_arc-sessions.with-ids.txt ppl_0.2_0.7_0.1
./ppl.sh ../../data/arc-sessions.with-ids.lm 0.2 0.8 0 ../../data/test_arc-sessions.with-ids.txt ppl_0.2_0.8_0
./ppl.sh ../../data/arc-sessions.with-ids.lm 1 0 0 ../../data/test_arc-sessions.with-ids.txt ppl_1_0_0
