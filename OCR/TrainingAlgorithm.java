import org.ejml.simple.SimpleMatrix;

/**
 * Represents a supervised training algorithm for a neural network.
 * @param <N> The type of NeuralNetwork his algorithm can be used on.
 */
public interface TrainingAlgorithm<N extends NeuralNetwork> {
	/**
	 * Called once at the top of the training data.
	 * If the number of training iterations is larger than the traingData List,
	 * this method will get called again each time the training data is exhausted
	 * and re-fed into this algorithm.
	 * @param nn
	 */
	default void startEpoch(N nn, TrainingData trainingData) { /* by default, do nothing */ }
	
	/**
	 * Updates the given NeuralNetwork based on this particular training algorithm.
	 * Called once for each expected, actual output pair.
	 * @param nn
	 * @param expectedOutput
	 * @param actualOutput
	 */
	void update(N nn, SimpleMatrix expectedOutput, SimpleMatrix actualOutput);
}
