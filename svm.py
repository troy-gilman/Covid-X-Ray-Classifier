import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics
import argparse
import plot


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

train_features = np.load(args['dataset'] + "/train_features.npy")
train_targets = np.load(args['dataset'] + "/train_targets.npy")
test_features = np.load(args['dataset'] + "/test_features.npy")[:100]
test_targets = np.load(args['dataset'] + "/test_targets.npy")[:100]


def run_svm(train_set, test_set):
    train_set_n = train_set[0].shape[0]
    test_set_n = test_set[0].shape[0]
    print('[INFO] Train set size:', train_set_n)
    print('[INFO] Test set size:', test_set_n)

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(kernel="poly", degree=6)
    classifier.fit(train_set[0], train_set[1])

    # Now predict the value of the digit on the second half:
    predicted = classifier.predict(test_set[0])

    # Report metrics
    # print(metrics.classification_report(test_set[1], predicted))
    cm = metrics.confusion_matrix(test_set[1], predicted)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    # show the confusion matrix, accuracy, sensitivity, and specificity
    # print(cm)
    print("[INFO] Accuracy: {:.4f}".format(acc))
    print("[INFO] Sensitivity: {:.4f}".format(sensitivity))
    print("[INFO] Specificity: {:.4f}".format(specificity))

    plot.plot_cm('plots/svm_' + str(train_set_n) + '_cm.png', train_set_n, cm)

    print()
    return acc


accuracies = {}
accuracies[20] = run_svm([train_features[:20], train_targets[:20]], [test_features, test_targets])
accuracies[50] = run_svm([train_features[20:70], train_targets[20:70]], [test_features, test_targets])
accuracies[200] = run_svm([train_features[70:270], train_targets[70:270]], [test_features, test_targets])
accuracies[500] = run_svm([train_features[270:770], train_targets[270:770]], [test_features, test_targets])

plot.plot_acc('plots/svm_acc_plot.png', 'SVM', accuracies)
