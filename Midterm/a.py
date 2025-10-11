import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt

# -- different numbers of nearest neighbors to test

# Fast (~1 min):
# ks = [*range(1,10), *range(10,100,10), *range(100,501,50)]

# Slow (~1 hr):
ks = list(range(1, 1001, 1))

# -- output image resolution
dpi = 1200  # e.g. 300 (publication quality)

class Scenario:
	def __init__(self, k: int):
		self.k = k
		self.accuracy: float = None  # as proportion [0.0, 1.0]
		self.real_time: float = None       # in seconds
		self.cpu_time: float = None        # in seconds

	def __str__(self):
		return str(self.k)

def load_mnist_csv(file):
	df = pd.read_csv(file)  # data frame
	X = df.iloc[:, 1:].values  # get all data except first column (labels) as numpy array
	Y = df.iloc[:, 0].values  # get first column (labels) as numpy array
	return (X / 255, Y)  # return labeled input-output pairs as tuple(X, Y)
	# Note:
	# 	X is row-major (i.e. each row is one feature set/vector) since
	# 	scikit-learn expects in this way. This is in contrast to a typical
	# 	backpropagation feedforward neural network where each feature set would
	# 	be a column, and the columns would be independent samples.

if __name__ == "__main__":
	# load data
	print("Loading data...")
	X_train, Y_train = load_mnist_csv("mnist_train.csv")  # training data, and labels
	X_test, Y_test = load_mnist_csv("mnist_test.csv")  # test data, and labels
	print()

	# test model for different K values
	scenarios = [Scenario(k) for k in ks]
	for scenario in scenarios:
		print(f"Testing with K={scenario.k}...")

		model = KNeighborsClassifier(n_neighbors = scenario.k)  # set up model
		model.fit(X_train, Y_train)  # lazy; no need to profile
		
		t0_real = time.perf_counter()  # profiling (start)
		t0_cpu = time.process_time()

		Y_predict = model.predict(X_test)  # run model on test data

		scenario.cpu_time = time.process_time() - t0_cpu  # profiling (end)
		scenario.real_time = time.perf_counter() - t0_real
		scenario.accuracy = accuracy_score(Y_test, Y_predict)
		# Note: order is important ^^

		print(f"  Accuracy: {scenario.accuracy:.2f}")
		print(f"  Real time: {scenario.real_time:.3f} s")
		print(f"  CPU time: {scenario.cpu_time:.3f} s")
		print()

	# plot results
	fig, ax1 = plt.subplots()
	ax1.plot(ks, [s.accuracy for s in scenarios], label='Accuracy', linestyle="-", color="red")
	ax1.set_xlabel('K (number of neighbors)')
	ax1.set_ylabel('Accuracy (proportion correct)', color="red")
	ax1.tick_params(axis='y', labelcolor="red")

	ax2 = ax1.twinx()
	ax2.plot(ks, [s.real_time for s in scenarios], label='Real Time', linestyle="-", color="grey")
	ax2.plot(ks, [s.cpu_time for s in scenarios], label='CPU Time', linestyle="-", color="blue")
	ax2.set_ylabel('Time (seconds)', color="blue")
	ax2.tick_params(axis='y', labelcolor="blue")

	# Combine legends from both axes
	lines_1, labels_1 = ax1.get_legend_handles_labels()
	lines_2, labels_2 = ax2.get_legend_handles_labels()
	ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

	plt.title('KNeighborsClassifier: Different K values')
	plt.tight_layout()  # automatically adjust the spacing between subplots to prevent overlap of labels, titles, and ticks
	fig.savefig('a-out.png', dpi=dpi, bbox_inches='tight')  # save the figure to a file
	plt.show()
