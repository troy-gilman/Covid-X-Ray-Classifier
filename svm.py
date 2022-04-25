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

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(test_set[1], predicted)))
    print()


run_svm((train_features[:5], train_targets[:5]), (test_features, test_targets))
run_svm((train_features[5:15], train_targets[5:15]), (test_features, test_targets))
run_svm((train_features[15:115], train_targets[15:115]), (test_features, test_targets))
run_svm((train_features[115:315], train_targets[115:315]), (test_features, test_targets))
