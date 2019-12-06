import sys
from subprocess import Popen, PIPE
from collections import defaultdict
from scipy.stats import describe
from itertools import combinations
from scipy import stats
import json
import jsonlines
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

def get_median_seed(seeds):
    return sorted([(seed, data['accuracy']) for seed, data in seeds.items()], key=lambda x: -x[1])[int(len(seeds)/2)][0]

def evaluate(model_path, test_path, cuda_device, output_path):
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

def predict(model_path, test_path, cuda_device, output_path):
        args = [
            'allennlp',
            'predict',
            model_path,
            test_path,
            "--include-package",
            "concite",
            "--predictor",
            "acl_classifier",
            "--cuda-device",
            str(cuda_device),
            "--use-dataset-reader",
            "--silent",
            "--output-file",
            output_path
        ]
        p = Popen(args, stdout=PIPE)
        for line in iter(p.stdout.readline, b''):
            sys.stdout.write(line)
        p.communicate()

def mcnemar(mask_a, mask_b):
    n1 = sum([int(b and not a) for a, b in zip(mask_a, mask_b)])
    n2 = sum([int(a and not b) for a, b in zip(mask_a, mask_b)])
    stat = ((abs(n1 - n2) -1) ** 2) / float(n1 + n2)
    pvalue = stats.chi2.sf(stat, 1)
    return stat, pvalue

if __name__ == "__main__":

    serialization_dir = sys.argv[1]
    test_path = sys.argv[2]
    cuda_device = sys.argv[3]
    output = sys.argv[4]

    model_paths = [os.path.join(dp, f)
            for dp, dn, filenames in os.walk(serialization_dir)
            for f in filenames if os.path.split(f)[-1] == 'model.tar.gz']

    model_metrics = defaultdict(dict)

    for model_path in model_paths:
        dir_path = os.path.split(model_path)[0]
        model_name = dir_path.split('/')[-1]
        seed = dir_path.split('/')[-2]
        output_path = os.path.join(dir_path, "eval_metrics.json")
        if not os.path.isfile(output_path):
            evaluate(model_path, test_path, cuda_device, output_path)
        model_metrics[model_name][seed] = summarize_metrics(output_path)

    for model, seeds in model_metrics.items():
        median_seed = get_median_seed(seeds)
        model_metrics[model]['median'] = model_metrics[model][median_seed]
        model_metrics[model]['median_seed'] = median_seed
        dir_path = os.path.join(serialization_dir, median_seed, model)
        model_path = os.path.join(dir_path, 'model.tar.gz')
        output_path = os.path.join(dir_path, "predictions.json")
        if not os.path.isfile(output_path):
            predict(model_path, test_path, cuda_device, output_path)
        with open(output_path) as f:
            data = list(jsonlines.Reader(f))
        y_true = [ex['label'] for ex in data]
        y_pred = [ex['pred_label'] for ex in data]

        correct_mask = [int(pred == true) for pred, true in zip(y_pred, y_true)]
        model_metrics[model]['correct_mask'] = correct_mask

    for a, b in combinations(model_metrics.keys(), 2):
        print("{} {}:".format(a, b))
        mcnemar_value = mcnemar(model_metrics[a]['correct_mask'], model_metrics[b]['correct_mask'])
        print(mcnemar_value)
        print('')
    
    with open(output, 'w') as f:
        json.dump(model_metrics, f)
