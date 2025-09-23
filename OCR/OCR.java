
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;

import javax.imageio.ImageIO;

import org.ejml.simple.SimpleMatrix;

public class OCR {
	// All input images need to be the same size
	public static final int IMG_WIDTH = 128;
	public static final int IMG_HEIGHT = 128;
	
	// Where we can load input data from
	public static final File IMG_DIR = new File("C:\\Users\\mathh\\Pictures\\NIST SD19\\by_field");
	
	public static double grey(BufferedImage img, int x, int y) {
		int rgb = img.getRGB(x, y);
		int blue = rgb & 0xFF;
		rgb >>>= 2;
		int green = rgb & 0xFF;
		rgb >>>= 2;
		int red = rgb & 0xFF;
		return (double) (red + green + blue) * (1.0 / 3.0);
	}
	
	public static double blue(BufferedImage img, int x, int y) {
		return (double) (img.getRGB(x, y) & 0xFF) * (1.0 / 0xFF);
	}
	
	/**
	 * Reads in the grey-scale image pixel data 
	 * @param file
	 * @param in
	 * @param col
	 * @throws IOException
	 */
	public static void readImg(File file, SimpleMatrix in, int col) throws IOException {
		BufferedImage img = ImageIO.read(file);
		for (int y = 0; y < IMG_HEIGHT; y++)
			for (int x = 0; x < IMG_WIDTH; x++) {
				
				// flatten the pixel data since the input vector must be a 1D column vector				
				int row = y * IMG_WIDTH + x;
				
				// assume grey-scale, and just read blue channel (lowest 8 bits)
				in.set(row, col, 1 - blue(img, x, y));  // blue (i.e. white): 0.0, black: 1.0
			}
	}

	public static void main(String[] args) {
		Path prefix = IMG_DIR.toPath();  // same as IMG_DIR, but type java.nio.file.Path
		
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
		NeuralNetwork nn = new NeuralNetwork(IMG_WIDTH * IMG_HEIGHT, 1008, categories.length);
		nn.randomize(rng::nextGaussian);  // randomize the weights
		
		// 2. Prepare Training Data:
		TrainingData trainingData = new TrainingData();
		
		final class IOPair implements TrainingDataPair {
			final String path;  // path (relative to prefix) to input .png file
			final int c;        // expected output neuron index (i.e. category)
			
			IOPair(String path, int c) {
				this.path = path;
				this.c = c;
			}
			
			@Override
			public void input(SimpleMatrix in, int col) throws IOException {
				readImg(new File(IMG_DIR, path), in, col);
			}
			
			@Override
			public void expectedOutput(SimpleMatrix out, int col) {
				// The expected output is a column vector of all 0's except in position c, the
				// category index.
				out.set(c, col, 1.0);
			}
			
			@Override
			public String toString() {
				return path;  // for debugging
			}
		}
		
		for (int n = 0; n <= 7; n++) {  // "hsf_{n}" sub-folder
			if (n == 5)  continue;  // no folder hsf_5 in NIST SD19
			
			File hsfDir = new File(IMG_DIR, "hsf_" + n);
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
					trainingData.add(new IOPair(prefix.relativize(file.toPath()).toString(), c));
			}
		}
		
		// 3. Train (@ learninRate, η (eta))
		double learningRate = 3.0;
		int batchSize = (int) Math.ceil(Math.sqrt(trainingData.size()));  // 753
		int epochs = 3000;
		TrainingAlgorithm algo = new StochasticGradientDescent(learningRate, rng);
		System.out.println("Batch Size: " + batchSize);
		nn.batchTrain(trainingData, epochs, batchSize, algo);
	}
}