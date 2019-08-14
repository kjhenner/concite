import sys
import random
import os
import json

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file) as f_in:
    with open(output_file, 'w') as f_out:
        for line in f_in.readlines():
            j = json.loads(line)
            if j.get('diabetes_label') and j.get('abstract'):
                f_out.write(line)
