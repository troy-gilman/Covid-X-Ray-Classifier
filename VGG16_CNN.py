# USAGE
# python VGG16_CNN.py --dataset datasets/kaggle                
# python VGG16_CNN.py --dataset datasets/covid-chestxray-dataset
# python VGG16_CNN.py --dataset datasets/combined

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
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="covid19.model",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 12
BS = 8
total_size = 50    #total size of 3 datasets resp 50,340,390
train_size =total_size*0.8
test_size=total_size*0.2

train_features = np.load(args['dataset'] + "/train_features.npy").reshape((-1, 224, 224, 3))[:train_size]
train_targets = np.load(args['dataset'] + "/train_targets.npy")[:train_size]
test_features = np.load(args['dataset'] + "/test_features.npy").reshape((-1, 224, 224, 3))[:test_size]
test_targets = np.load(args['dataset'] + "/test_targets.npy")[:test_size]

# train_features = np.repeat(train_features[:, :, np.newaxis], 3, -1).reshape((-1, 224, 224, 3))
# test_features = np.repeat(test_features[:, :, np.newaxis], 3, -1).reshape((-1, 224, 224, 3))

# perform one-hot encoding on the labels
lb = LabelBinarizer()
train_targets = lb.fit_transform(train_targets)
train_targets = to_categorical(train_targets)
test_targets = lb.fit_transform(test_targets)
test_targets = to_categorical(test_targets)

# load the VGG16 network, ensuring the head FC layer sets are left off
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

##summerize model
print(model.summary())

##visualize the model 
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	trainAug.flow(train_features, train_targets, batch_size=BS),
	steps_per_epoch=len(train_features) // BS,
	validation_data=(test_features, test_targets),
	validation_steps=len(test_features) // BS,
	epochs=EPOCHS)


# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(test_features, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(test_targets.argmax(axis=1), predIdxs))

# compute the confusion matrix and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(test_targets.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# plot the training loss and accuracy
N = EPOCHS + 1
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(1, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(1, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(1, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(1, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# serialize the model to disk
print("[INFO] saving COVID-19 detector model...")
model.save(args["model"], save_format="h5")
