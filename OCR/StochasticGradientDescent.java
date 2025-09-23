import java.util.Random;
import java.util.function.Function;

import org.ejml.simple.SimpleMatrix;

public class StochasticGradientDescent implements TrainingAlgorithm {

	public final double learningRate;
	public final Random rng;
	
	/**
	 * Calculates the cost between (actualOutput - expectedOutput).
	 * 
	 * The matrices may by column vectors for online training, or matrices
	 * representing many expected-actual output pairs for batch training. In the
	 * later case, this function is expected to return the arithmetic average of the
	 * cost of each column (i.e. expected-actual output) pair.
	 */
	public Function<SimpleMatrix, Double> costFunction = m -> 0.5 * Util.meanSqNorm(m);
	
	/**
	 * Each column is a vector function of partial derivatives, each with respect to
	 * a change in the <code>i<code>th activation (i.e. the final activation output
	 * from each output neuron on the final layer).
	 */
	public Function<SimpleMatrix, SimpleMatrix> costGradient = m -> m;
	
	
	public StochasticGradientDescent(double learningRate, Random rng) {
		this.learningRate = learningRate;
		this.rng = rng;
	}
	
	@Override
	public void epochStart(NeuralNetwork nn, TrainingData trainingData) {
		trainingData.shuffle(rng);
	}
	
	@Override
	public void update(NeuralNetwork nn, SimpleMatrix outputDeltas, SimpleMatrix[] a, SimpleMatrix[] z, int batchSize) {
		// gradient decent:
		SimpleMatrix gradient = costGradient.apply(outputDeltas); // the (local) gradient of the cost function with respect to the current layer's neuron activations
		
		// back propagation
		for (int t = nn.T - 1; t >= 0; t--) {
			Util.apply(z[t], nn.activationDerivatives[t]);
			Util.elementMult(gradient, z[t]);  // i.e. the error for this layer
			SimpleMatrix delta = gradient;  // This copy is import since we are about to update gradient for the next layer, but need this delta for updating the weights and biases.
			gradient = nn.weights[t].transpose().mult(delta);  // now the (local) gradient foe the precedent layer
			
			Util.scale(-learningRate / batchSize, delta);
			Util.addEquals(nn.biases[t], Util.horizontalSum(delta)); // Update nn.biases[t] based on delta
			Util.addEquals(nn.weights[t], delta.mult(a[t]).transpose()); // Update nn.wieghts[t] based on delta
		}
	}
}
