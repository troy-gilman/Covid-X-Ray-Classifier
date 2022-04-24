import os
import numpy as np
from random import shuffle
import cv2
import matplotlib.pyplot as plt

IMG_RES = (224, 224)
DATASET = "covid-chestxray-dataset"
IMG_PATH = "images/" + DATASET + "/"
DATASET_PATH = "datasets/" + DATASET + "/"
CLASSES = {0: 'normal', 1: 'covid'}

features = []
targets = []

# Load images and their class into features and targets
for c in CLASSES.keys():
    for file in os.listdir(IMG_PATH + CLASSES[c]):
        image = cv2.imread(IMG_PATH + CLASSES[c] + "/" + str(file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, IMG_RES)
        features.append(image)
        targets.append(c)

n_samples = len(features)
test_size = int(n_samples / 10)
train_size = n_samples - test_size
print("Size of dataset: ", n_samples)

# Shuffle features and targets
features_shuf = []
targets_shuf = []
index_shuf = list(range(n_samples))
shuffle(index_shuf)
for i in index_shuf:
    features_shuf.append(features[i])
    targets_shuf.append(targets[i])

# Final arrays
test_features = np.array(features_shuf[:test_size]).reshape((test_size, -1)) / 255.0
test_targets = np.array(targets_shuf[:test_size])
train_features = np.array(features_shuf[test_size:]).reshape((train_size, -1)) / 255.0
train_targets = np.array(targets_shuf[test_size:])
print("Size of test dataset: ", test_features.shape[0])
print("Size of train dataset: ", train_features.shape[0])

# Save arrays
np.save(DATASET_PATH + "test_features", test_features)
np.save(DATASET_PATH + "test_targets", test_targets)
np.save(DATASET_PATH + "train_features", train_features)
np.save(DATASET_PATH + "train_targets", train_targets)

# plt.imshow(test_features[0].reshape(IMAGE_RES))
# plt.show()