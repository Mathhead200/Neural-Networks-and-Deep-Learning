
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;
import java.util.function.BiFunction;

import javax.imageio.ImageIO;

import org.ejml.simple.SimpleMatrix;

public class OCR {
	public static SimpleMatrix read(File file) throws IOException {
		BufferedImage img = ImageIO.read(file);
		int width = img.getWidth();
		int height = img.getHeight();
		SimpleMatrix v = new SimpleMatrix(Math.multiplyExact(width, height), 1);  // column vector
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++) {
				int rgb = img.getRGB(x, y);
				int blue = rgb & 0xFF;
				rgb >>>= 2;
				int green = rgb & 0xFF;
				rgb >>>= 2;
				int red = rgb & 0xFF;
				double grey = (double) (red + green + blue) / 3.0;
				v.set(y * width + x, 1 - grey / 0xFF);  // white: 0.0, black: 1.0
			}
		return v;
	}

	public static void main(String[] args) {
		// Where we can load input data from
		File prefix = new File("C:\\Users\\mathh\\Pictures\\NIST SD19\\by_field");
		Path prefixPath = prefix.toPath();
		
		String[] categories = new String[10 + 26 + 26]; // { '0', ..., '9', 'A', ..., 'Z', 'a', ..., 'z' }
		for (char digit = '0'; digit <= '9'; digit++)
			categories[digit - '0'] = "digit\\" + Integer.toHexString((int) digit);
		for (char upper = 'A'; upper <= 'Z'; upper++)
			categories[10 + (upper - 'A')] = "upper\\" + Integer.toHexString((int) upper);
		for (char lower = 'a'; lower <= 'z'; lower++)
			categories[10 + 26 + (lower -'a')] = "lower\\" + Integer.toHexString((int) lower);
		
		Random rng = new Random();
		
		// 1. Initialize Model:
		// The number of sigmoid-neurons in each layer.
		// (The first layer is the input layer.)
		// (The last layer is the output layer.)
		NeuralNetwork nn = new NeuralNetwork(128 * 128, 1008, categories.length);
		nn.randomize(rng);  // randomize the weights
		
		// 2. Prepare Training Data:
		CategorizedTrainingData trainingData = new CategorizedTrainingData(nn);
		
		final class IOPair implements TrainingDataPair {
			final String path;  // path (relative to prefix) to input .png file
			final int c;        // expected output neuron index (i.e. category)
			
			IOPair(String path, int c) {
				this.path = path;
				this.c = c;
			}
			
			@Override
			public SimpleMatrix input() throws IOException {
				return read(new File(prefix, path));
			}
			
			@Override
			public SimpleMatrix expectedOutput() {
				return trainingData.expectedOutputs[c];
			}
			
			@Override
			public String toString() {
				return path;  // for debugging
			}
		}
		
		for (int n = 0; n <= 7; n++) {  // "hsf_{n}" sub-folder
			if (n == 5)  continue;  // no folder hsf_5 in NIST SD19
			
			File hsfDir = new File(prefix, "hsf_" + n);
			if (!hsfDir.isDirectory()) {
				System.err.println("Missing training data directory: " + hsfDir.getPath());
				continue;  // skip
			}
			
			for (int c = 0; c < categories.length; c++) {
				File cDir = new File(hsfDir, categories[c]);
				if (!cDir.isDirectory()) {
					System.err.println("Missing training data directory: " + cDir.getPath());
					continue;  // skip
				}
				
				for (File file : cDir.listFiles())
					trainingData.add(new IOPair(prefixPath.relativize(file.toPath()).toString(), c));
			}
		}
		
		// 3. Train (@ learninRate, η (eta))
		double learningRate = 3.0;
		int batchSize = (int) Math.ceil(Math.sqrt(trainingData.size()));  // 753
		int epochs = 3000;
		System.out.println("Batch Size: " + batchSize);
		nn.train(trainingData, epochs * trainingData.size(), new StochasticGradientDescent(
				learningRate, batchSize, rng, Util::distanceSq));
	}
}