import sys

source_path = sys.argv[1]
target_path = sys.argv[2]
output_path = sys.argv[3]

line_lengths = []
vocab = set()

with open(source_path) as f:
    for line in f.readlines():
        tokens = line.split()
        for token in tokens:
            vocab.add(token)
        line_lengths.append(len(tokens))

with open(target_path) as f:
    lines = [line.split() for line in f.readlines()]

for i, line in enumerate(lines):
    lines[i] = lines[i][:line_lengths[i % len(line_lengths)]]

lines = 

with open(output_path, 'w') as f:
    for i, line in enumerate(lines):
        f.write(' '.join(lines[i][:line_lengths[i % len(line_lengths)]]) + '\n')
