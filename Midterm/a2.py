import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from a import ks, dpi, load_mnist_csv

def binary_classifier_output(Y, no=0, yes=1):
	""" Y filtered into either yes or no matches for each label
		e.g. If Y = [K,   A,   D,   A,   B,   R,   A  ], then
			return dict({
				A:  [no,  yes, no,  yes, no,  no,  yes],
				B:  [no,  no,  no,  no,  yes, no,  no ],
				D:  [no,  no,  yes, no,  no,  no,  no ],
				K:  [yes, no,  no,  no,  no,  no,  no ],
				R:  [no,  no,  no,  no,  no,  yes, no ]
			})
	"""
	# Note: set(Y) removes duplicates: set of unique labels only
	return { label: [yes if y == label else no for y in Y] for label in set(Y) }

def decide(y, threshold = 0.5, no=0, yes=1):
	return yes if y >= threshold else no

D = range(10)  # labels are numbers (0-9) (i.e. digits)

class Scenario:
	def __init__(self, k: int):
		self.k = k
		# for each label
		self.mse        = { L: None for L in D }  # should be on interval [0.0, 1.0] (b.c. max error is 1, min is 0)
		self.accuracy   = { L: None for L in D }  # as proportions on interval [0.0, 1.0]
		self.metrics    = { L: None for L in D }  # as five proportions (PRECISION, RECALL, TYPE_1, TYPE_2, F1)
		self.real_time  = { L: None for L in D }  # in seconds
		self.cpu_time   = { L: None for L in D }  # in seconds

	def __str__(self):
		return str(self.k)

def metrics(Y_actual, Y_predict, no=0, yes=1):
	"""	Returns tuple(precision, recall, type_1_error_rate, type_2_error_rate)

		[0]: Precision
		i.e out of all positive predictions, what proportion were correctly classified as positive?

		[1]: Recall (sensitivity)
		i.e. out of all positive cases, what proportion were correctly classified as positive?

		[2]: Type I error rate (false positive rate)
		i.e. out of all negative cases, what proportion were incorrectly classified as positive?

		[3]: Type II error rate (false negative rate)
		i.e. out of all positive cases, what proportion were incorrectly classified as negative?

		[4]: F1 score
		i.e. harmonic mean of precision and recall
	"""
	# partition negative and positive cases
	z = zip(Y_actual, Y_predict)
	predicted_pos_cases = []
	actual_pos_cases = []
	actual_neg_cases = []
	for actual, predict in z:
		if actual == yes:
			actual_pos_cases.append((actual, predict))
		elif actual == no:
			actual_neg_cases.append((actual, predict))
		else:
			raise ValueError(f"Invalid actual value {actual}; expected {no} or {yes}.")

		if predict == yes:
			predicted_pos_cases.append((actual, predict))
		elif predict != no:
			raise ValueError(f"Invalid predicted value {predict}; expected {no} or {yes}.")
	
	# count predictions and errors
	precision_count = sum(1 for actual, predict in predicted_pos_cases if actual == predict)
	recall_count = sum(1 for actual, predict in actual_pos_cases if actual == predict)
	type1_errors = sum(1 for actual, predict in actual_neg_cases if actual != predict)
	# type2_errors = sum(1 for actual, predict in actual_pos_cases if actual != predict)
	
	# compute rates/proportions if possible
	precision        = precision_count / len(predicted_pos_cases)      if len(predicted_pos_cases) != 0 else None
	recall           = recall_count / len(actual_pos_cases)            if len(actual_pos_cases) != 0 else None
	type1_error_rate = type1_errors / len(actual_neg_cases)            if len(actual_neg_cases) != 0 else None
	type2_error_rate = 1 - recall                                      if recall is not None else None
	f1               = 2 * (precision * recall) / (precision + recall) if precision is not None and recall is not None and (precision + recall) != 0 else None

	return (precision, recall, type1_error_rate, type2_error_rate, f1)

# enum
PRECISION = 0
RECALL = 1
TYPE_1 = 2
TYPE_2 = 3
F1 = 4

def plot(scenarios, L, dpi=300):
	# CSS color chart/reference: https://davidbau.com/colors
	fig, ax1 = plt.subplots()
	ax1.plot(ks, [s.accuracy[L]           for s in scenarios], label="Accuracy",          linestyle="-",  color="red")
	ax1.plot(ks, [s.mse[L]                for s in scenarios], label="MSE",               linestyle="-", color="darkorange")
	ax1.plot(ks, [s.metrics[L][PRECISION] for s in scenarios], label="Precision",         linestyle="--",  color="mediumseagreen")
	ax1.plot(ks, [s.metrics[L][RECALL]    for s in scenarios], label="Recall",            linestyle=":", color="lightseagreen")
	ax1.plot(ks, [s.metrics[L][F1]        for s in scenarios], label="F1 Score",          linestyle="-.", color="midnightblue")
	ax1.plot(ks, [s.metrics[L][TYPE_1]    for s in scenarios], label="Type 1 Error Rate", linestyle="-.",  color="purple")
	ax1.plot(ks, [s.metrics[L][TYPE_2]    for s in scenarios], label="Type 2 Error Rate", linestyle=":",  color="indigo")
	ax1.set_xlabel("K (number of neighbors)")
	ax1.set_ylabel("Metrics (proportion)", color="red")
	ax1.tick_params(axis="y", labelcolor="red")

	ax2 = ax1.twinx()
	ax2.plot(ks, [s.cpu_time[L]  for s in scenarios], label="CPU Time",  linestyle="-",  color="mediumblue")
	ax2.plot(ks, [s.real_time[L] for s in scenarios], label="Real Time", linestyle="--", color="dimgray")
	ax2.set_ylabel("Time (seconds)", color="mediumblue")
	ax2.tick_params(axis="y", labelcolor="mediumblue")

	# Combine legends from both axes
	lines_1, labels_1 = ax1.get_legend_handles_labels()
	lines_2, labels_2 = ax2.get_legend_handles_labels()
	ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

	plt.title(f"KNeighborsRegressor for digit L={L}: Different K values")
	plt.tight_layout()  # automatically adjust the spacing between subplots to prevent overlap of labels, titles, and ticks
	fig.savefig(f"out/a2-out,{L}.png", dpi=dpi, bbox_inches="tight")  # save the figure to a file
	plt.show()

def stats(xs):
	if len(xs) == 0:
		return (None, None, None, None)
	
	min = xs[0]
	max = xs[0]
	mean = 0
	std_dev = 0

	for x in xs:
		if   x < min:  min = x
		elif x > max:  max = x
		mean += x
	mean /= len(xs)
	
	for x in xs:
		d = x - mean
		std_dev = d*d
	std_dev = np.sqrt(std_dev / len(xs))

	return (mean, std_dev, min, max)

# enum
MEAN=0
STD_DEV=1
MIN=2
MAX=3

if __name__ == "__main__":
	# load data
	print("Loading data...")
	X_train, Y_train = load_mnist_csv("mnist_train.csv")  # training data, and labels
	Y_train_binary = binary_classifier_output(Y_train)
	
	X_test, Y_test = load_mnist_csv("mnist_test.csv")  # test data, and labels
	Y_test_binary = binary_classifier_output(Y_test)
	print()

	# test model for different K values
	scenarios = [Scenario(k) for k in ks]
	for scenario in scenarios:
		print(f"Testing with K={scenario.k}...")

		for L in D:
			model = KNeighborsRegressor(n_neighbors = scenario.k)  # set up model
			model.fit(X_train, Y_train_binary[L])  # lazy; no need to profile
			
			t0_real = time.perf_counter()  # profiling (start)
			t0_cpu = time.process_time()

			Y_raw_predict = model.predict(X_test)  # run model on test data to get continuous predictions
			# print(f"[{L}]:", [float(x) for x in sorted(set(Y_raw_predict))])  # DEBUG
			Y_realized_predict = [decide(y) for y in Y_raw_predict]  # descretize to binary predictions

			scenario.cpu_time[L] = time.process_time() - t0_cpu  # profiling (end)
			scenario.real_time[L] = time.perf_counter() - t0_real
			scenario.mse[L] = mean_squared_error(Y_test_binary[L], Y_raw_predict)
			scenario.accuracy[L] = accuracy_score(Y_test_binary[L], Y_realized_predict)
			scenario.metrics[L] = metrics(Y_test_binary[L], Y_realized_predict)
			# Note: order is important ^^ b.c. profiling

		# print log
		def print_stats(field_name: str, field: dict, idx: int=None, p=5, indent=2, width=24):
			values = [x for _, x in field.items()]
			num_dots = width - indent - len(field_name) - 3

			s = stats(values if idx is None else [v[idx] for v in values])
			print(f"{indent * " "}{field_name}: {num_dots * "."} rng=[{s[MIN]:.{p}f}, {s[MAX]:.{p}f}], mean={s[MEAN]:.{p}f}, std_dev={s[STD_DEV]:.{p}f}")
		
		print_stats("Accuracy", scenario.accuracy)
		print_stats("Mean squared error", scenario.mse)
		print_stats("Precision", scenario.metrics, idx=PRECISION)
		print_stats("Recall", scenario.metrics, idx=RECALL)
		print_stats("Type 1 error rate", scenario.metrics, idx=TYPE_1)
		print_stats("Type 2 error rate", scenario.metrics, idx=TYPE_2)
		print_stats("F1 score", scenario.metrics, idx=F1)
		print_stats("Real time (in s)", scenario.real_time)
		print_stats("CPU time (in s)", scenario.cpu_time)
		print()

	# output CSV
	print("Saving results as CSV...")
	pd.DataFrame([{
		"K": scenario.k,
		"L": L,
		"Accuracy": scenario.accuracy[L],
		"Mean-squared error": scenario.mse[L],
		"Precision": scenario.metrics[L][PRECISION],
		"Recall": scenario.metrics[L][RECALL],
		"Type 1 error rate": scenario.metrics[L][TYPE_1],
		"Type 2 error rate": scenario.metrics[L][TYPE_2],
		"F1 score": scenario.metrics[L][F1],
		"Real time (s)": scenario.real_time[L],
		"CPU time (s)": scenario.cpu_time[L]
	} for scenario in scenarios for L in D]).to_csv("out/a2-out.csv", index=False)

	# plot
	for L in D:
		plot(scenarios, L, dpi=dpi)
