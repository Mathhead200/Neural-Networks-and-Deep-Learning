import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Iterator;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

import org.ejml.simple.SimpleMatrix;

public class NeuralNetwork {
	public final SimpleMatrix[] weights;
	public final SimpleMatrix[] biases;
	
	// The activation functions and their derivatives for each transition between layers
	public Function<Double, Double>[] activationFunctions;
	// Inputs are both z and sigmoid(z) since both are computed and available when this function is called
	public BiFunction<Double, Double, Double>[] activationDerivatives;
	
	/** Number of transitions/transformations between layers. */
	public final int T;
	
	@SuppressWarnings("unchecked")
	public NeuralNetwork(int... layerSizes) {
		T = layerSizes.length - 1;
		
		weights = new SimpleMatrix[T];
		biases = new SimpleMatrix[T];
		activationFunctions = (Function<Double, Double>[]) new Function[T];
		activationDerivatives = (BiFunction<Double, Double, Double>[]) new Function[T];
		for (int t = 0; t < T; t++) {
			weights[t] = new SimpleMatrix(layerSizes[t + 1], layerSizes[t]);
			biases[t] = new SimpleMatrix(layerSizes[t + 1], 1);
			activationFunctions[t] = Util::sigmoid;
			activationDerivatives[t] = (_, s) -> Util.dSigmoidImplicit(s);
		}
		
	}
	
	public int inDim() {
		return weights[0].getNumCols();
	}
	
	public int outDim() {
		return weights[T - 1].getNumRows();
	}
	
	public int layers() {
		return T + 1;
	}
	
	public void randomize(Supplier<Double> next) {
		for (SimpleMatrix w : weights)
			Util.apply(w, next);
		
		for (SimpleMatrix b : biases)
			Util.apply(b, next);
	}
	
	public void batchTrain(TrainingData trainingData, int epochs, int maxBatchSize, TrainingAlgorithm algo) {
		final int IN_DIM = inDim();
		final int OUT_DIM = outDim();
		final int L = layers();
		
		algo.trainingStart(this);
		
		for (int epoch = 0; epoch < epochs; epoch++) {
			algo.epochStart(this, trainingData);
			
			Iterator<TrainingDataPair> iter = trainingData.iterator();
			BATCH: while(true) {  // exhaust iter
				
				// 1. Build a batch.
				SimpleMatrix inputs = new SimpleMatrix(IN_DIM, maxBatchSize);
				SimpleMatrix expectedOutputs = new SimpleMatrix(OUT_DIM, maxBatchSize);
				
				int batchSize;
				for (batchSize = 0; batchSize < maxBatchSize; batchSize++) {  // i.e., column "j"
					if (!iter.hasNext()) {
						// If true, there is no more training data in the iter. End the epoch.
						if (batchSize == 0)
							break BATCH;
						// Otherwise, this is a residue/tail batch.
						// Allow NN to update on this truncated batch.
						break;
					}
					
					TrainingDataPair pair = null;
					try {
						// add training data pair to batch
						pair = iter.next();
						pair.input(inputs, batchSize);  // batchSize is also the current column "j"
						pair.expectedOutput(expectedOutputs, batchSize);
					
					} catch(Exception ex) {
						System.err.println("WARNING: Truncating batch becasue Error getting input or expected output from: " + pair);
						ex.printStackTrace();
						
						// This exception is thrown before batchSize++ (for loop), so the current value
						// of batchSize is correct: i.e. all data before batchSize is valid.

						// Optimization: no reason to update NN if their is no training data in this
						// batch. Instead, skip the update. Instead start building the next batch.
						if (batchSize == 0)
							continue BATCH;
						
						// Otherwise, allow the NN update.
					}
				}

				// 2. Transform our input layer [0], through all hidden layers, into our output
				//    layer [L = T+1], calculating activation matrices `a`, and pre-activation
				//    matrices `z` along the way.
				SimpleMatrix[] a = new SimpleMatrix[L];  // Activation matrices for each layer. Size L = T + 1.
				SimpleMatrix[] z = new SimpleMatrix[T];  // The pre-activation weighted input for each transition;
					// e.g. z[0] is the linear output of layer 0 (the input layer) which is fed into layer 1 as
					// weighted (pre-activation) input before applying the activation function (e.g. sigmoid).
				
				a[0] = inputs;  // the activations for layer [0] (i.e. the input neurons) are the inputs
				for (int t = 0; t < T; t++) {
					SimpleMatrix w = weights[t];  // weights for this transformation/transition between layers t -> t + 1
					SimpleMatrix b = biases[t];   // biases for this transformation/transition between layers t -> t + 1
					z[t] = Util.broadcast(w.mult(a[t]), b);  // calculate linear outputs for layer t (i.e. pre-activation inputs for layer t + 1
					Util.apply(a[t + 1] = z[t].copy(), activationFunctions[t]);  // apply vectorized activation function to calculate the activation matrix for layer t + 1
				}
				
				// 3. Update weights and biases based on difference from expected output.
				SimpleMatrix outputDeltas = a[T].minus(expectedOutputs);
				algo.update(this, outputDeltas, a, z, batchSize);
				
				// 4. DEBUG: log cost of each batch to make sure it's going down
				if (algo instanceof StochasticGradientDescent sgd)
					System.out.printf("[epoch: %04d, batchSize: %04d] Cost: %f%n", epoch, batchSize, sgd.costFunction.apply(outputDeltas));
				else
					System.out.println("DEBUG: Unrecognizeed algo. No cost info.");
			}
		}
	}
	
	private static interface Save {
		void save(OutputStream out, NeuralNetwork nn) throws IOException;
	}
	
	private static interface Load {
		NeuralNetwork load(InputStream in) throws IOException;
	}
	
	public static enum IOFormat {
		@SuppressWarnings("unused")
		BINARY((out, nn) -> {
			throw new IOException("TODO: Unimplemented format: BINARY"); // TODO: stub
		}, in -> {
			throw new IOException("TODO: Unimplemented format: BINARY"); // TODO: stub
		}),
		
		@SuppressWarnings("unused")
		JSON((out, nn) -> {
			throw new IOException("TODO: Unimplemented format: JSON"); // TODO: stub
		}, in -> {
			throw new IOException("TODO: Unimplemented format: JSON"); // TODO: stub
		}),
		
		@SuppressWarnings("unused")
		TEXT((out, nn) -> {
			throw new IOException("TODO: Unimplemented format: TEXT"); // TODO: stub
		}, in -> {
			throw new IOException("TODO: Unimplemented format: TEXT"); // TODO: stub
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
