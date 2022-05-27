# Covid X-Ray Classifier

## About
This project was built by Troy Gilman and Rohit Rajput for the course *Machine Learning/Pattern Recognition* at Northeastern University. The full report for this project can be found at `Covid-Classifier.pdf`.

## Instructions

### Dependencies
- numpy
- opencv-python
- tqdm
- sklearn
- matplotlib
- tensorflow

### Build Dataset
The complete dataset of X-ray images used for training the classifier models is included in this repo. To build the raw images into a usable dataset, execute the following script.

> $ python3 src/build_dataset.py --images images

### Run Support Vector Machine (SVM)
To fit the SVM model to the dataset and validate the performance, run the following script.

> $ python3 src/svm.py

### Run Convolutional Neural Network (CNN)
To train the CNN model on the dataset and validate the performance, run the following script.

> $ python3 src/cnn.py



