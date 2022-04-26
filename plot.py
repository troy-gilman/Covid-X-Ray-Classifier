import matplotlib.pyplot as plt


def plot_epochs(file_path, n, epochs, loss, val_loss, accuracy, val_accuracy):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.plot(epochs, accuracy, label="train_acc")
    plt.plot(epochs, val_accuracy, label="val_acc")
    plt.title("Training/Validation Loss and Accuracy N = " + str(n))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.ylim([0.0, 1.0])
    plt.legend(loc="lower left")
    plt.savefig(file_path)


def plot_cm(file_path, n, cm):
    plt.figure()
    plt.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions')
    plt.ylabel('Actuals')
    plt.title('Confusion Matrix N = ' + str(n))
    plt.gca().grid(False)
    plt.savefig(file_path)


def plot_acc(file_path, model_type, accuracies):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(accuracies.keys(), accuracies.values())
    plt.scatter(accuracies.keys(), accuracies.values(), label="val_acc")
    plt.title("Validation Accuracy for " + model_type)
    plt.xlabel("# Training Samples")
    plt.ylabel("Accuracy")
    plt.ylim([0.0, 1.0])
    plt.legend(loc="upper left")
    plt.savefig(file_path)
