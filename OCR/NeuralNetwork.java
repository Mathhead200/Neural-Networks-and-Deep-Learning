import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Iterator;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

import org.ejml.simple.SimpleMatrix;

public class NeuralNetwork {
	public final SimpleMatrix[] weights;
	public final SimpleMatrix[] biases;
	
	/** Number of transitions/transformations between layers. */
	public final int T;
	
	public NeuralNetwork(int... layerSizes) {
		T = layerSizes.length - 1;
		
		weights = new SimpleMatrix[T];
		for (int t = 0; t < T; t++)
			weights[t] = new SimpleMatrix(layerSizes[t + 1], layerSizes[t]);
		
		biases = new SimpleMatrix[T];
		for (int t = 0; t < T; t++)
			biases[t] = new SimpleMatrix(layerSizes[t + 1], 1);
	}
	
	public void randomize(Random rng) {
		final Function<Double, Double> gaussian = x -> rng.nextGaussian();
		
		for (SimpleMatrix w : weights)
			Util.apply(w, gaussian);
		
		for (SimpleMatrix b : biases)
			Util.apply(b, gaussian);
	}
	
	public void train(TrainingData trainingData, int n, TrainingAlgorithm<? super NeuralNetwork> algo) {
		Iterator<TrainingDataPair> iter = null;
		for (int i = 0; i < n; i++) {
			if (i == 0 || !iter.hasNext()) {
				algo.startEpoch(null, trainingData);  // e.g. for StochasticGradientDescent, this (re-)shuffles the data set
				iter = trainingData.iterator();
			}
			
			var pair = iter.next();	
			try {
				SimpleMatrix v = pair.input();  // get input vector for this file
				
				// transform our input layer, through all hidden layers, into our output layer
				for (int t = 0; t < T; t++) {
					SimpleMatrix w = weights[t];
					SimpleMatrix b = biases[t];
					v = w.mult(v).plus(b);
					Util.apply(v, Util::sigmoid);  // vectorize
				}
				
				// update weights and biases based on difference from expected output
				algo.update(this, pair.expectedOutput(), v);
				// TODO: use gradient decent to update the model
				System.out.println(pair + " : " + algo);  // DEBUG
				
			} catch(Exception ex) {
				System.err.println(ex);
			}
		}
	}
	
	public void train(TrainingData trainingData, TrainingAlgorithm<? super NeuralNetwork> algo) {
		train(trainingData, trainingData.size(), algo);
	}
	
	private static interface Save {
		void save(OutputStream out, NeuralNetwork nn) throws IOException;
	}
	
	private static interface Load {
		NeuralNetwork load(InputStream in) throws IOException;
	}
	
	public static enum IOFormat {
		BINARY((out, nn) -> {
			throw new IOException("TODO: Unimplemented format: BINARY"); // TODO: stub
		}, in -> {
			throw new IOException("TODO: Unimplemented format: BINARY"); // TODO: stub
		}),
		
		JSON((out, nn) -> {
			throw new IOException("TODO: Unimplemented format: JSON"); // TODO: stub
		}, in -> {
			throw new IOException("TODO: Unimplemented format: JSON"); // TODO: stub
		}),
		
		TEXT((out, nn) -> {
			// TODO: stub
		}, in -> {
			return null; // TODO: stub
		});
		
		public final Save save;
		public final Load load;
		
		private IOFormat(Save save, Load load) {
			this.save = save;
			this.load= load;
		}
	}
	
	public void save(OutputStream out, IOFormat format) throws IOException {
		format.save.save(out, this);
	}
	
	public static NeuralNetwork load(InputStream in, IOFormat format) throws IOException {
		return format.load.load(in);
	}
}
