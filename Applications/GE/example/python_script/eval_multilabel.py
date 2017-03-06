
import sys
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

def load_embedding_matrix(path):
    with open(path) as in_file:
        header = in_file.readline()
        args = header.strip().split()
        num_lines, embedding_size = int(args[0]), int(args[1])
        X = np.zeros((num_lines, embedding_size))
        for line in in_file:
            args = line.strip().split()
            node = int(args[0])
            X[node, :] = map(lambda x: float(x), args[1:])
        return X
    return X

def load_labels(path):
    nodes = []
    Y = []
    with open(path) as in_file:
        for line in in_file:
            args = line.strip().split()
            node = int(args[0])
            labels = map(lambda x: int(x), args[1:])
            nodes.append(node)
            Y.append(labels)
    Y = MultiLabelBinarizer().fit_transform(Y)
    return nodes, Y

def eval_score(full_X, nodes, Y, proportion=0.1, run_times=3):
    X = np.zeros((len(nodes), full_X.shape[1]))
    for i, node in enumerate(nodes):
        X[i, :] = full_X[node, :]

    sum_micro, sum_macro = .0, .0
    for t in range(run_times):
        X, Y = shuffle(X, Y)
        train_size = int(X.shape[0] * proportion)
        X_train, Y_train = X[:train_size], Y[:train_size]
        X_dev, Y_dev = X[train_size:], Y[train_size:]

        clf = OneVsRestClassifier(LogisticRegression())
        clf.fit(X_train, Y_train)

        # rank probability, choose the largest K as labels
        Y_pred = np.zeros_like(Y_dev, dtype=int)
        Y_prob = clf.predict_proba(X_dev)
        for i in range(Y_prob.shape[0]):
            num_truth = np.nonzero(Y_dev[i, :])[0].shape[0]
            if num_truth > 0:
                labels_pred = np.argsort(-Y_prob[i, :])[:num_truth]
                Y_pred[i, labels_pred] = 1

        sum_micro += f1_score(Y_dev, Y_pred, average='micro')
        sum_macro += f1_score(Y_dev, Y_pred, average='macro')
    return sum_micro / run_times, sum_macro / run_times

if __name__ == "__main__":
    embed_path = sys.argv[1]
    label_path = sys.argv[2]
    proportion = float(sys.argv[3])

    X = load_embedding_matrix(embed_path)
    print 'loaded embedding matrix'

    nodes, Y = load_labels(label_path)
    print 'loaded multi labels, number of samples %d' % len(nodes)

    micro, macro = eval_score(X, nodes, Y, proportion)
    print 'proportion %f, micro %f, macro %f' % (proportion, micro, macro)
