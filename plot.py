import matplotlib.pyplot as plt


def plot_results(train_losses,train_acc,test_losses,test_acc):
	plt.plot(train_losses, label='Training loss')
	plt.plot(test_losses, label='Validation loss')
	plt.legend(frameon=False)
	plt.show()

	plt.plot(train_acc, label='Training accuracy')
	plt.plot(test_acc, label='Validation accuracy')
	plt.legend(frameon=False)
	plt.show()

	plt.plot(train_losses, label='Training loss')
	plt.legend(frameon=False)
	plt.show()

	plt.plot(test_losses, label='Validation loss')
	plt.legend(frameon=False)
	plt.show()

	plt.plot(train_acc, label='Training accuracy')
	plt.legend(frameon=False)
	plt.show()

	plt.plot(test_acc, label='Validation accuracy')
	plt.legend(frameon=False)
	plt.show()