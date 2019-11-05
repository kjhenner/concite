import sys
import json
import os

def to_latex_row(data, title):
    return ' & '.join(["\\textsc{" + title + "}",
        str(data['best_validation_average_F1']),
        str(data['best_validation_macro_F1']),
        str(data['best_validation_accuracy'])
    ])

def macro_average(data):
    keys = [key for key in data.keys() if key[:15] == 'best_validation' and key[-2:] == 'F1']
    return sum([data[key] for key in keys]) / len(keys)

paths = [os.path.join(dp, f)
        for dp, dn, filenames in os.walk('/shared-1/projects/concite/serialization')
        for f in filenames if os.path.splitext(f)[0] == 'metrics']

for path in paths:
    with open(path) as f:
        data = json.load(f)
        data['best_validation_macro_F1'] = macro_average(data)
        title = path.split('/')[-2]
        print(to_latex_row(data, title))
