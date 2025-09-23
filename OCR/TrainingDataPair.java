import org.ejml.simple.SimpleMatrix;

public interface TrainingDataPair {
	/**
	 * Get the input vector for this training data pair.
	 * 
	 * Postcondition: The input vector should be stored in the given column of the
	 * given matrix.
	 * 
	 * Precondition: It can be assumed for the sake of optimization that all
	 * elements of the given column are initialized to <code>0.0<code>.
	 * 
	 * @param in  - Where the input vector should be stored.
	 * @param col - The index of the column where the input vector should be stored.
	 * @throws Exception - May through an exception if the input data can not be
	 *                   accessed on demand for any reason. In this case, the batch
	 *                   containing this data pair may be skipped.
	 */
	void input(SimpleMatrix in, int col) throws Exception;
	
	/**
	 * Get the expected output vector for this training data pair.
	 * 
	 * Postcondition: The expected output vector should be stored in the given
	 * column of the given matrix.
	 * 
	 * Precondition: It can be assumed for the sake of optimization that all
	 * elements of the given column are initialized to <code>0.0</code>.
	 * 
	 * @param out  - Where the expected output vector should be stored.
	 * @param col - The index of the column where the expected output vector should
	 *            be stored.
	 * @throws Exception - May through an exception if the expected output data can
	 *                   not be accessed on demand for any reason. In this case, the
	 *                   batch containing this data pair may be skipped.
	 */
	void expectedOutput(SimpleMatrix out, int col) throws Exception;
	
	/**
	 * Helper function that creates a column vector, stores the input.
	 * 
	 * @see #input(SimpleMatrix, int)
	 * @param dim - dimensions of the vector
	 * @return Newly created vector containing input
	 * @throws Exception - If input is not available
	 */
	default SimpleMatrix input(int dim) throws Exception {
		SimpleMatrix in = new SimpleMatrix(dim, 1);
		input(in, 0);
		return in;
	}
	
	/**
	 * Helper function that creates a column vector, stores the expected output.
	 * 
	 * @see #output(SimpleMatrix, int)
	 * @param dim - dimensions of the vector
	 * @return Newly created vector containing expected output
	 * @throws Exception - If expected output is not available
	 */
	default SimpleMatrix expectedOutput(int dim) throws Exception {
		SimpleMatrix out = new SimpleMatrix(dim, 1);
		expectedOutput(out, 0);
		return out;
	}
}
