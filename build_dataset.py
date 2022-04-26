import os
import numpy as np
from random import shuffle
import cv2
from tqdm import tqdm

TEST_SPLIT = 0.2
IMG_RES = (224, 224)
DATASETS = ["kaggle"]
IMG_PATH = "images/"
BUILD_PATH = "datasets/"
CLASSES = {0: 'normal', 1: 'covid', 2: 'pneumonia'}


def balance_classes(class_features, class_targets):
    features = []
    targets = []
    min_class_count = min([len(class_x) for class_x in class_features.values()])
    for i in class_features.keys():
        print('Samples removed from class', CLASSES[i] + ':', len(class_features[i][min_class_count:]))
        features += class_features[i][:min_class_count]
        targets += class_targets[i][:min_class_count]
    return features, targets


def build_dataset(dataset_name):
    print('Building dataset from', dataset_name)

    # Load images and their class into features and targets
    class_features = {0: [], 1: [], 2: []}
    class_targets = {0: [], 1: [], 2: []}
    for c in CLASSES.keys():
        print('Building class:', CLASSES[c])
        dir_path = IMG_PATH + dataset_name + "/" + CLASSES[c]
        for file in tqdm(os.listdir(dir_path)):
            image = cv2.imread(dir_path + "/" + str(file))
            if image is not None:
                image = cv2.resize(image, IMG_RES)
                class_features[c].append(image)
                class_targets[c].append(c)

    features, targets = balance_classes(class_features, class_targets)

    n_samples = len(features)
    test_size = int(n_samples * TEST_SPLIT)
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
    print("Size of test set: ", test_features.shape[0])
    print("Size of train set: ", train_features.shape[0])
    print()

    # Save arrays
    np.save(BUILD_PATH + dataset_name + "/test_features", test_features)
    np.save(BUILD_PATH + dataset_name + "/test_targets", test_targets)
    np.save(BUILD_PATH + dataset_name + "/train_features", train_features)
    np.save(BUILD_PATH + dataset_name + "/train_targets", train_targets)


if __name__ == "__main__":
    for name in DATASETS:
        build_dataset(name)
