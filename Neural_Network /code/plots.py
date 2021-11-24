import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
	ax1.plot(train_losses, label='Training Loss')
	ax1.plot(valid_losses, label='Validation Loss')
	ax1.set_title('Loss Learning Curves')
	ax1.legend(loc='upper right')
	ax1.set(xlabel='Epoch', ylabel='Loss')

	ax2.plot(train_accuracies, label='Training Accuracy')
	ax2.plot(valid_accuracies, label='Validation Accuracy')
	ax2.set_title('Accuracy Learning Curves')
	ax2.legend(loc='upper left')
	ax2.set(xlabel='Epoch', ylabel='Accuracy')

	fig.savefig('./learning_curves.png')
	pass


def plot_confusion_matrix(results, class_names):
	y_true = [x[0] for x in results]
	y_pred = [x[1] for x in results]
	cm = confusion_matrix(y_true, y_pred, normalize='true')
	cm = np.around(cm, decimals=4)
	cm_display = ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap=plt.cm.Blues, xticks_rotation=30 ,values_format='g')
	plt.title('Normalized Confusion Matrix')
	plt.savefig('./fig/Confusion_Matrix.png')

	pass
