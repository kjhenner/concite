import sys
import jsonlines
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

if __name__ == "__main__":
    input_path = sys.argv[1]
    out_path = sys.argv[2]
    with open(input_path) as f:
        data = list(jsonlines.Reader(f))

    title="Confusion Matrix"
    y_true = [ex['label'] for ex in data]
    y_pred = [ex['pred_label'] for ex in data]
    labels = list(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(dpi=300, figsize=(8, 8))
    cmap=plt.cm.Blues
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(out_path)
