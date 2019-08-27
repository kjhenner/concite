import sys
import random
import json
import os

input_file = sys.argv[1]
train_prop = float(sys.argv[2])

with open(input_file) as f:
    lines = f.readlines()

lines = [line for line in lines if json.loads(line).get('abstract') and len(json.loads(line).get('graph_vector'))]

random.shuffle(lines) 
line_count = len(lines)

train_par = int(line_count * train_prop)
test_par = int(line_count/2 * (1-train_prop))

print(line_count)
print(train_par)
print(test_par)

dirname, basename = os.path.split(input_file)

with open(os.path.join(dirname, 'train_'+basename), 'w') as f:
    f.writelines(lines[:train_par])

with open(os.path.join(dirname, 'test_'+basename), 'w') as f:
    f.writelines(lines[train_par:-test_par])

with open(os.path.join(dirname, 'dev_'+basename), 'w') as f:
    f.writelines(lines[-test_par:])
