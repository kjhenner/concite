import sys
from subprocess import Popen, PIPE
from collections import defaultdict
from scipy.stats import describe
import json
import os
import re

def to_latex_row(data, title):
    return ' & '.join(["\\textsc{" + title + "}",
        "{:.2f}".format(data['P']),
        "{:.2f}".format(data['R']),
        "{:.2f}".format(data['F1']),
        "{:.2f}".format(data['accuracy'])
    ])

def summarize_metrics(path):
    with open(path) as f:
        data = json.load(f)
    P = list(map(float, [data[k]
        for k in data.keys() if re.match(r'.*P$', k)]))
    R = list(map(float, [data[k]
        for k in data.keys() if re.match(r'.*R$', k)]))
    return {
            'P': sum(P) / float(len(P)),
            'R': sum(R) / float(len(R)),
            'F1': data['average_F1'],
            'accuracy': data['accuracy']
    }

if __name__ == "__main__":

    serialization_dir = sys.argv[1]
    test_path = sys.argv[2]
    cuda_device = sys.argv[3]
    output = sys.argv[4]

    model_paths = [os.path.join(dp, f)
            for dp, dn, filenames in os.walk(serialization_dir)
            for f in filenames if os.path.split(f)[-1] == 'model.tar.gz']
    
    model_metrics = defaultdict(list)
    for model_path in model_paths:
        dir_path = os.path.split(model_path)[0]
        model_name = dir_path.split('/')[-1]
        seed = dir_path.split('/')[-2]
        output_path = os.path.join(dir_path, "eval_metrics.json")
        args = [
            'allennlp',
            'evaluate',
            model_path,
            test_path,
            "--include-package",
            "concite",
            "--cuda-device",
            str(cuda_device),
            "--output-file",
            output_path
        ]
        p = Popen(args, stdout=PIPE)
        for line in iter(p.stdout.readline, b''):
            sys.stdout.write(line)
        p.communicate()
        model_metrics[model_name].append(summarize_metrics(output_path))
    summary_metrics = {}
    for model, seeds in model_metrics.items():
        results = defaultdict(list)
        for seed in seeds:
            for metric, value in seed.items():
                results[metric].append(value)
        for metric, value in results.items():
            results[metric] = describe(value)
        summary_metrics[model] = results
    with open(output, 'w') as f:
        json.dump(summary_metrics, f)
