import sys
import os
import jsonlines
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    vector_path = sys.argv[3]
    label_field = sys.argv[4]
    serialization_dir = sys.argv[5]

    vectorizer = TfidfVectorizer()

    with open(train_path) as f:
        train_examples = [ex for ex in jsonlines.Reader(f)]

    with open(test_path) as f:
        test_examples = [ex for ex in jsonlines.Reader(f)]

    with open(vector_path) as f:
        if vector_path.split('.')[-1] == 'jsonl':
            vector_lookup = {ex['paper_id']: ex['tfidf']
                    for ex in jsonlines.Reader(f)}
        else:
            vector_lookup = {line.split()[0] : list(map(float, line.split()[1:]))
                    for line in f.readlines()}

    labels = list(set([ex[label_field] for ex in train_examples]))
    int_to_label = dict(enumerate(labels))
    label_to_int = {label: i for i, label in enumerate(labels)}

    train_examples = [ex for ex in train_examples if vector_lookup.get(ex['paper_id'])]
    test_examples = [ex for ex in test_examples if vector_lookup.get(ex['paper_id'])]

    train_X = [vector_lookup[ex['paper_id']] for ex in train_examples if vector_lookup.get]
    test_X = [vector_lookup[ex['paper_id']] for ex in test_examples]

    train_Y = [label_to_int[ex[label_field]] for ex in train_examples]
    test_Y = [label_to_int[ex[label_field]] for ex in test_examples]

    try:
        os.mkdir(serialization_dir)
    except FileExistsError:
        pass

    for seed in [666, 669, 672, 675, 678]:
        clf = make_pipeline(StandardScaler(), SVC(probability=True, kernel='sigmoid', random_state=seed, gamma='scale', C=0.9))
        clf.fit(train_X, train_Y)

        pred_Y = clf.predict(test_X)

        report = classification_report(test_Y, pred_Y, target_names = labels, digits=4)

        with open(os.path.join(serialization_dir, '{}_smc_report'.format(seed)), 'w') as f:
            f.write(report)
        

        with open(os.path.join(serialization_dir, '{}_smc_predictions.tsv'.format(seed)), 'w') as f:
            f.write('\t'.join(labels)+"\n")
            for pair in zip(test_Y, pred_Y):
                f.write("{}\t{}\n".format(*pair))
