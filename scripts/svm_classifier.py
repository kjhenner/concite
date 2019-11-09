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
    tfidf_path = sys.argv[3]
    serialization_dir = sys.argv[4]
    vectorizer = TfidfVectorizer()

    with open(train_path) as f:
        train_examples = [ex for ex in jsonlines.Reader(f)]

    with open(test_path) as f:
        test_examples = [ex for ex in jsonlines.Reader(f)]

    with open(tfidf_path) as f:
        tfidf_lookup = {ex['paper_id']: ex['tfidf'] for ex in jsonlines.Reader(f)}

    labels = list(set([ex['combined_workshop'] for ex in train_examples]))
    int_to_label = dict(enumerate(labels))
    label_to_int = {label: i for i, label in enumerate(labels)}

    train_X = [tfidf_lookup[ex['paper_id']] for ex in train_examples]
    test_X = [tfidf_lookup[ex['paper_id']] for ex in test_examples]

    train_Y = [label_to_int[ex['combined_workshop']] for ex in train_examples]
    test_Y = [label_to_int[ex['combined_workshop']] for ex in test_examples]

    clf = make_pipeline(StandardScaler(), SVC(probability=True))
    clf.fit(train_X, train_Y)

    pred_Y = clf.predict(test_X)
    print(classification_report(test_Y, pred_Y, target_names = labels))
    
    try:
        os.mkdir(serialization_dir)
    except FileExistsError:
        pass

    with open(os.path.join(serialization_dir, 'smc_model.pickle'), 'wb') as f:
        pickle.dump(clf, f)

    with open(os.path.join(serialization_dir, 'smc_model_predictions.tsv'), 'w') as f:
        f.write('\t'.join(labels)+"\n")
        for pair in zip(test_Y, pred_Y):
            f.write("{}\t{}\n".format(*pair))

