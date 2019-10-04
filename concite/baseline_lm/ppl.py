#!/bin/python3
import sys
import math
from collections import defaultdict

lm_path = sys.argv[1]
l1 = float(sys.argv[2])
l2 = float(sys.argv[3])
l3 = float(sys.argv[4])
test_data_path = sys.argv[5]
output_path = sys.argv[6]

def get_counts(string):
    counts = string.split()[2:]
    type_count = counts[0].split("=")[1]
    token_count = counts[1].split("=")[1]
    return (type_count, token_count)

def load_lm(path):
    with open(path) as f:
        lines = f.readlines()
    counts = [{},{},{}] 
    n = 0
    lm = [{} for _ in range(0,3)]
    for line in lines[4:]:
        if len(line) > 1:
            if line[0] == '\\':
                n += 1
            else:
                lm[n-1][tuple(line.split()[3:])] = (tuple(line.split()[0:3]))
    return (counts, lm)


counts, lm = load_lm(lm_path)

oov_num = 0
p_sum = 0
word_num = 0

with open(test_data_path) as f:
    sentences = [s.split() for s in f.readlines()]

sent_num = len(sentences)

out = []
for i, tokens in enumerate(sentences):
    out.append('Sent #' + str(i+1) + ': ' + ' '.join(['<s>'] + tokens + ['</s>']))
    word_num += len(tokens)
    sent_oovs = 0
    sent_p_sum = 0
    for j, token in enumerate(tokens + ['</s>']):
        note = ''
        if j - 1 < 0:
            wi_1 = '<s>'
        else:
            wi_1 = tokens[j-1]
        if j-2 < 0:
            wi_2 = '<s>'
        else:
            wi_2 = tokens[j-2]
        context_str = wi_2 + ' ' + wi_1
        if lm[0].get((token,)):
            p1 = l1 * float(lm[0][(token,)][1])
            if lm[1].get((wi_1, token)):
                p2 = l2 * float(lm[1][(wi_1, token)][1])
            else:
                note = ' (unseen ngrams)'
                p2 = 0.0
            if lm[2].get((wi_2, wi_1, token)):
                p3 = l3 * float(lm[2][(wi_2, wi_1, token)][1])
            else:
                note = ' (unseen ngrams)'
                p3 = 0.0
            prob = p1 + p2 + p3
            log_prob = math.log(prob, 10)
            sent_p_sum += log_prob
            p_sum += log_prob
        else:
            note = ' (unknown word)'
            prob = 0
            log_prob = - float('inf')
            sent_oovs += 1
            oov_num += 1
        out.append("{}:lg P({} | {}) = {:.10f}{}".format(j+1, token, context_str, log_prob, note))
    out.append("1 sentence, {} words, {} OOVs".format(len(tokens), sent_oovs))
    sent_cnt = len(tokens) + 1 - sent_oovs
    sent_total = (-1.0 * sent_p_sum) / float(sent_cnt)
    sent_ppl = 10.0 ** sent_total
    out.append("lgprob={:.10f} ppl={:.10f}".format(sent_p_sum, sent_ppl))

out.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
out.append("sent_num={} word_num={}, oov_num={}".format(sent_num, word_num, oov_num))
cnt = word_num + len(sentences) - oov_num
total = (-1 * p_sum) / cnt
ppl = 10.0 ** total
out.append("logprob={:.10f} ave_logprob={:.10f} ppl={:.10f}".format(p_sum, p_sum/word_num, ppl))

with open(output_path, 'w') as f:
    f.write("\n".join(out))
