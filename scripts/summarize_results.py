import sys
import json
import os
import re

def to_latex_row(data, title):
    P = list(map(float, [data[k]
        for k in data.keys() if re.match(r'.*P', k)]))
    mean_P = sum(P) / float(len(P))
    R = list(map(float, [data[k]
        for k in data.keys() if re.match(r'.*R', k)]))
    mean_R = sum(R) / float(len(R))
    return ' & '.join(["\\textsc{" + title + "}",
        "{:.4f}".format(mean_P),
        "{:.4f}".format(mean_R),
        "{:.4f}".format(data['average_F1']),
        "{:.4f}".format(data['accuracy'])
    ])

#paths = [os.path.join(dp, f)
#        for dp, dn, filenames in os.walk(sys.argv[1])
#        for f in filenames if os.path.splitext(f)[-1] == 'metrics.json']

paths = [os.path.join(dp, f)
        for dp, dn, filenames in os.walk(sys.argv[1])
        for f in filenames]

for path in paths:
    with open(path) as f:
        data = json.load(f)
        title = path.split('/')[-1]
        print(to_latex_row(data, title))
