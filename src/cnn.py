# USAGE
# python cnn.py --dataset [dataset path]

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import plot

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='datasets', help="path to input dataset")
ap.add_argument("-m", "--model", type=str, default="covid19.model", help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 15
BS = 10

train_features = np.load(args['dataset'] + "/train_features.npy").reshape((-1, 224, 224, 3))
train_targets = np.load(args['dataset'] + "/train_targets.npy")
test_features = np.load(args['dataset'] + "/test_features.npy").reshape((-1, 224, 224, 3))[:100]
test_targets = np.load(args['dataset'] + "/test_targets.npy")[:100]


def run_cnn(train_set, test_set):
	train_set_n = train_set[0].shape[0]
	test_set_n = test_set[0].shape[0]

	# perform one-hot encoding on the labels
	lb = LabelBinarizer()
	train_set[1] = lb.fit_transform(train_set[1])
	test_set[1] = lb.fit_transform(test_set[1])

	# load the VGG16 network, ensuring the head FC layer sets are left off
	base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

	# construct the head of the model that will be placed on top of the base model
	head_model = base_model.output
	head_model = AveragePooling2D(pool_size=(4, 4))(head_model)
	head_model = Flatten(name="flatten")(head_model)
	head_model = Dense(64, activation="relu")(head_model)
	head_model = Dropout(0.5)(head_model)
	head_model = Dense(3, activation="softmax")(head_model)

	# place the head FC model on top of the base model (this will become
	# the actual model we will train)
	model = Model(inputs=base_model.input, outputs=head_model)

	# loop over all layers in the base model and freeze them so they will
	# *not* be updated during the first training process
	for layer in base_model.layers:
		layer.trainable = False

	# compile our model
	print("[INFO] Compiling model...")
	opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)

	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

	# visualize the model
	# plot_model(model, to_file='media/model_diagram.png', show_shapes=True, show_layer_names=True)

	# initialize the training data augmentation object
	train_aug = ImageDataGenerator(
		rotation_range=15,
		fill_mode="nearest")

	print('[INFO] Train set size:', train_set_n)
	print('[INFO] Test set size:', test_set_n)

	# train the head of the network
	print("[INFO] Training head...")
	H = model.fit(
		train_aug.flow(train_set[0], train_set[1], batch_size=BS),
		steps_per_epoch=train_set_n // BS,
		validation_data=(test_set[0], test_set[1]),
		validation_steps=test_set_n // BS,
		epochs=EPOCHS)

	# make predictions on the testing set
	print("[INFO] Evaluating network...")
	pred_targets = model.predict(test_set[0], batch_size=BS)

	# for each image in the testing set we need to find the index of the
	# label with corresponding largest predicted probability
	pred_classes = np.argmax(pred_targets, axis=1)
	actual_classes = np.argmax(test_set[1], axis=1)

	# show a nicely formatted classification report
	# print(classification_report(actual_classes, pred_classes))

	# compute the confusion matrix and use it to derive the raw
	# accuracy, sensitivity, and specificity
	cm = confusion_matrix(actual_classes, pred_classes)
	total = sum(sum(cm))
	acc = (cm[0, 0] + cm[1, 1] + cm[2, 2]) / total
	sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
	specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

	# show the confusion matrix, accuracy, sensitivity, and specificity
	print(cm)
	print("[INFO] Accuracy: {:.4f}".format(acc))
	print("[INFO] Sensitivity: {:.4f}".format(sensitivity))
	print("[INFO] Specificity: {:.4f}".format(specificity))

	# plot the training loss and accuracy
	plot.plot_epochs(
		'media/cnn_' + str(train_set_n) + '_plot.png',
		train_set_n,
		np.arange(1, EPOCHS + 1),
		H.history["loss"],
		H.history["val_loss"],
		H.history["accuracy"],
		H.history["val_accuracy"])

	plot.plot_cm('media/cnn_' + str(train_set_n) + '_cm.png', train_set_n, cm)

	# serialize the model to disk
	# print("[INFO] Saving COVID-19 detector model...")
	# model.save(str(train_set_n) + '_' + args["model"], save_format="h5")
	print()
	return acc


accuracies = {}
accuracies[20] = run_cnn([train_features[:20], train_targets[:20]], [test_features, test_targets])
accuracies[50] = run_cnn([train_features[20:70], train_targets[20:70]], [test_features, test_targets])
accuracies[200] = run_cnn([train_features[70:270], train_targets[70:270]], [test_features, test_targets])
accuracies[500] = run_cnn([train_features[270:770], train_targets[270:770]], [test_features, test_targets])

plot.plot_acc('media/cnn_acc_plot.png', 'CNN', accuracies)
