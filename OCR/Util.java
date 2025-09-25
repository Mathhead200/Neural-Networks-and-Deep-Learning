import java.util.function.Function;
import java.util.function.Supplier;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;


public final class Util {
	public static void apply(SimpleMatrix m, Function<Double, Double> f) {
		final double[] data = ((DMatrixRMaj) m.getMatrix()).data;
		for (int i = 0; i < data.length; i++)
			data[i] = f.apply(data[i]);
	}
	
	public static void apply(SimpleMatrix m, Supplier<Double> f) {
		apply(m, _ -> f.get());
	}
	
	/**
	 * Adds m += v where m is an NxM matrix, and v is an Nx1 vector which is
	 * "broadcast" horizontally and added to each column. m will be modified, but v
	 * will not.
	 * 
	 * @param m - NxM matrix
	 * @param v - Nx1 column vector
	 * @return The modified input matrix, <code>m<code>
	 */
	public static SimpleMatrix broadcast(SimpleMatrix m, SimpleMatrix v) {
		DMatrixRMaj mRaw = m.getMatrix();
		int rows = mRaw.numRows;
		int cols = mRaw.numCols;
		double[] mData = mRaw.data;
		double[] vData = ((DMatrixRMaj) v.getMatrix()).data;
		
		for (int i = 0; i < rows; i++) {
			int offset = i * cols;
			for (int j = 0; j < cols; j++)
				mData[offset + j] += vData[i];
		}
		return m;
	}
	
	public static double sq(double x) {
		return x * x;
	}
	
	public static double sigmoid(double z) {
		return 1.0 / (1.0 + Math.exp(-z));
	}
	
	public static double dSigmoidExplicit(double z) {
//		double exp = Math.exp(-z);
//		return exp / sq(1 + exp);
		double s = sigmoid(z);
		return s * (1 - s);
	}
	
	public static double dSigmoidImplicit(double sigmoid) {
		return sigmoid * (1 - sigmoid);
	}
	
	/**
	 * Computes the sum of each column, and returns them as a row vector.
	 * 
	 * @param m Any matrix
	 * @return a row vector, where each element [i] contains the sum of column [i]
	 *         in matrix m.
	 */
	public static SimpleMatrix verticalSums(SimpleMatrix m) {
		DMatrixRMaj raw = m.getMatrix();
		int rows = raw.numRows;
		int cols = raw.numCols;
		double[] data = raw.data;
		
		double[] sums = new double[cols];
		for (int i = 0; i < rows; i++) {
			int offset = i * cols;
			for (int j = 0; j < cols; j++)
				sums[j] += data[offset + j];
		}
		return new SimpleMatrix(1, cols, true, sums);
	}
	
	/** @see #verticalSums(SimpleMatrix) */
	public static SimpleMatrix horizontalSum(SimpleMatrix m) {
		DMatrixRMaj raw = m.getMatrix();
		int rows = raw.numRows;
		int cols = raw.numCols;
		double[] data = raw.data;
		
		double[] sums =new double[rows];
		for (int i = 0; i < rows; i++) {
			int offset = i * cols;
			for (int j = 0; j < cols; j++)
				sums[i] += data[offset + j];
		}
		return new SimpleMatrix(rows, 1, true, sums);
	}
	
	public static double mean(double... values) {
		double sum = 0;
		for (double ele : values)
			sum += ele;
		return sum / values.length;
	}
	
	public static double mean(SimpleMatrix m) {
		return mean(((DMatrixRMaj) m.getMatrix()).data);
	}
	
	/**
	 * <p>
	 * Calculates the square of the Euclidian norm for each column, resulting in
	 * <code>n</code> distances where <code>n</code> is the number of columns in
	 * matrix <code>m</code>. Then averages those <code>n</code> norms using an
	 * arithmetic mean.
	 * </p>
	 * 
	 * @param m
	 * @return <code>(1.0 / n) * {sum of (|u[i] - v[i]|**2) for each column i}<code>
	 */
	public static double meanSqNorm(SimpleMatrix m) {
		return mean(verticalSums(m.elementPower(2)));
	}
	
	/**
	 * In-place. Modifies <code>a</code>
	 * @param a
	 * @param b
	 * @return <code>a</code>
	 */
	public static SimpleMatrix elementMult(SimpleMatrix a, SimpleMatrix b) {
		CommonOps_DDRM.elementMult(a.getMatrix(), b.getMatrix());
		return a;
	}
	
	/** @see CommonOps_DDRM#scale(double, DMatrixRMaj) */
	public static SimpleMatrix scale(double alpha, SimpleMatrix a) {
		CommonOps_DDRM.scale(alpha, a.getMatrix());
		return a;
	}
	
	/** @see CommonOps_DDRM#addEquals(DMatrixRMaj, DMatrixRMaj) */
	public static SimpleMatrix addEquals(SimpleMatrix a, SimpleMatrix b) {
		CommonOps_DDRM.addEquals(a.getMatrix(), b.getMatrix());
		return a;
	}
}
