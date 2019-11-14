import sys
import os
import jsonlines
import numpy as np
from allennlp.common.util import sanitize
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    input_path = sys.argv[1]
    vectorizer = TfidfVectorizer(stop_words='english', min_df=0.01)

    with open(input_path) as f:
        examples = [ex for ex in jsonlines.Reader(f) if ex['abstract']]

    X = np.asarray(vectorizer.fit_transform([ex['abstract'] for ex in examples]).todense())

    pathname, basename = os.path.split(input_path)
    with open(os.path.join(pathname, 'tfidf_' + basename), 'w') as f:
        writer = jsonlines.Writer(f)
        for i, ex in enumerate(examples):
            writer.write({'paper_id': ex['paper_id'], 'tfidf': list(X[i])})
