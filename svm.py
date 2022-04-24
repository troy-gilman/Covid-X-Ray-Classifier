import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics

dataset_path = "datasets/kaggle/"

train_features = np.load(dataset_path + "train_features.npy")
train_targets = np.load(dataset_path + "train_targets.npy")
test_features = np.load(dataset_path + "test_features.npy")
test_targets = np.load(dataset_path + "test_targets.npy")

# Create a classifier: a support vector classifier
classifier = svm.SVC(kernel="poly", degree=6)
classifier.fit(train_features, train_targets)

# Now predict the value of the digit on the second half:
predicted = classifier.predict(test_features)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(test_targets, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_targets, predicted))
