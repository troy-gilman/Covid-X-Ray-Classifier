import os
import numpy as np
from random import shuffle
import cv2

IMG_RES = (224, 224)
DATASETS = ["kaggle", "covid-chestxray-dataset"]
IMG_PATH = "images/"
BUILD_PATH = "datasets/"
CLASSES = {0: 'normal', 1: 'covid'}


def build_dataset(dataset_name="combined"):
    features = []
    targets = []

    if dataset_name == "combined":
        dataset_list = DATASETS
    else:
        dataset_list = [dataset_name]

    # Load images and their class into features and targets
    for dataset in dataset_list:
        for c in CLASSES.keys():
            dir_path = IMG_PATH + dataset + "/" + CLASSES[c]
            for file in os.listdir(dir_path):
                image = cv2.imread(dir_path + "/" + str(file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, IMG_RES)
                features.append(image)
                targets.append(c)

    n_samples = len(features)
    test_size = int(n_samples / 10)
    train_size = n_samples - test_size
    print("Size of dataset " + dataset_name + ":", n_samples)

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
    print()

    # Save arrays
    np.save(BUILD_PATH + dataset_name + "/test_features", test_features)
    np.save(BUILD_PATH + dataset_name + "/test_targets", test_targets)
    np.save(BUILD_PATH + dataset_name + "/train_features", train_features)
    np.save(BUILD_PATH + dataset_name + "/train_targets", train_targets)


if __name__ == "__main__":
    for name in DATASETS:
        build_dataset(dataset_name=name)
    build_dataset()
