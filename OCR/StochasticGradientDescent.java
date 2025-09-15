import java.util.Random;
import java.util.function.BiFunction;

import org.ejml.simple.SimpleMatrix;

public class StochasticGradientDescent implements TrainingAlgorithm<NeuralNetwork> {

	public final double learningRate;
	public final int batchSize;
	public final Random rng;
	public final BiFunction<SimpleMatrix, SimpleMatrix, Double> error;
	
	private int i = 0;
	private double accumulatedCost = 0;
	
	public StochasticGradientDescent(double learningRate, int batchSize, Random rng, BiFunction<SimpleMatrix, SimpleMatrix, Double> error) {
		this.learningRate = learningRate;
		this.batchSize = batchSize;
		this.rng = rng;
		this.error = error;
	}
	
	@Override
	public void startEpoch(NeuralNetwork nn, TrainingData trainingData) {
		trainingData.shuffle(rng);
	}
	
	@Override
	public void update(NeuralNetwork nn, SimpleMatrix expectedOutput, SimpleMatrix actualOutput) {
		accumulatedCost += error.apply(expectedOutput, actualOutput);
		if (++i >= batchSize) {
			accumulatedCost /= 2 * batchSize;  // half of the mean error
			
			// TODO: update weights
			
			i = 0;
			accumulatedCost = 0;
		}
	}

	@Override
	public String toString() {
		return "" + accumulatedCost;  // DEBUG
	}
}
