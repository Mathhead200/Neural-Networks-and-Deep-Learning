import org.ejml.simple.SimpleMatrix;

/**
 * Represents a supervised training algorithm for a neural network.
 * 
 * @param <N> The type of NeuralNetwork his algorithm can be used on.
 */
public interface TrainingAlgorithm {
	/**
	 * Called exactly once when training first begins.
	 * 
	 * @param nn
	 */
	default void trainingStart(NeuralNetwork nn) {
		/* by default, do nothing */ }

	/**
	 * Called once at the top of the training data.
	 * 
	 * @param nn
	 */
	default void epochStart(NeuralNetwork nn, TrainingData trainingData) {
		/* by default, do nothing */ }

	/**
	 * <p> Updates the given NeuralNetwork based on this particular training algorithm.
	 * Called once for each expected-actual output pair or batch. </p>
	 * 
	 * <p> If <code>batchSize > 1</code>, each expected-actual output pair is stored as
	 * a column in their respective matrices. </p>
	 *
	 * <p> Either matrix may have extraneous uninitialized columns (usually because of
	 * truncated batches). Only columns with an index between <code>0</code>
	 * (inclusive) and <code>batchSize</code> (exclusive) should be considered. </p>
	 * 
	 * <p> Pre-conndition: For optimization, it is safe to assume that
	 * <code>batchSize >= 1</code>. </p>
	 * 
	 * @param nn
	 * @param outputDeltas <code>actualOutput - expectedOutput<code>
	 * @param a Activations for each layer
	 * @param z Pre-activation weighted inputs (i.e. linear outputs) for each level.
	 * @param batchSize
	 */
	void update(NeuralNetwork nn, SimpleMatrix outputDeltas, SimpleMatrix[] a, SimpleMatrix[] z, int batchSize);
}
