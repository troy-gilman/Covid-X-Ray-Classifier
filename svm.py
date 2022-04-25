import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics

dataset_path = "datasets/combined/"

train_features = np.load(dataset_path + "train_features.npy")
train_targets = np.load(dataset_path + "train_targets.npy")
test_features = np.load(dataset_path + "test_features.npy")
test_targets = np.load(dataset_path + "test_targets.npy")


def run_svm(train_set, test_set):
    print("Size of train dataset: ", train_set[0].shape[0])
    print("Size of test dataset: ", test_set[0].shape[0])

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
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))
    print()


run_svm((train_features[:5], train_targets[:5]), (test_features, test_targets))
run_svm((train_features[5:15], train_targets[5:15]), (test_features, test_targets))
run_svm((train_features[15:115], train_targets[15:115]), (test_features, test_targets))
run_svm((train_features[115:315], train_targets[115:315]), (test_features, test_targets))
