import sys
from subprocess import Popen, PIPE
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

if __name__ == "__main__":

    serialization_dir = sys.argv[1]
    cuda_device = sys.argv[2]

    model_paths = [os.path.join(dp, f)
            for dp, dn, filenames in os.walk(serialization_dir)
            for f in filenames if os.path.split(f)[-1] == 'model.tar.gz']

    for model_path in model_paths:
        dir_path = os.path.split(model_path)[0]
        model_name = dir_path.split('/')[-1]
        seed = dir_path.split('/')[-2]
        args = [
            model_path,
            "--include-package concite",
            "--cuda-device {}".format(cuda_device),
            "--output-file {}".format(os.path.join(dir_path, "eval_metrics.json"))
        ]
        p = Popen(args, stdout=PIPE)
        for line in iter(p.stdout.readline, b''):
            sys.stdout.write(line)
        p.communicate()
