import org.ejml.simple.SimpleMatrix;

@SuppressWarnings("serial")
public class CategorizedTrainingData extends TrainingData {
	public final int categories;
	
	/**
	 * Associated expected output for each category.
	 * Each category gets its own output neuron.
	 */
	public final SimpleMatrix[] expectedOutputs;
	
	public CategorizedTrainingData(NeuralNetwork nn) {
		final int outputSize = nn.weights[nn.weights.length - 1].getNumRows();
		
		this.categories = outputSize;
		
		this.expectedOutputs = new SimpleMatrix[outputSize];
		for (int i = 0; i < this.expectedOutputs.length; i++) {
			SimpleMatrix v = new SimpleMatrix(outputSize, 1);
			v.set(i, 0, 1.0);
			expectedOutputs[i] = v;
		}
	}
}
