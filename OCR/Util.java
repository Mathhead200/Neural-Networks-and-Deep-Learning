import java.util.function.Function;

import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;


public final class Util {
	public static void apply(SimpleMatrix m, Function<Double, Double> f) {
		final double[] data = ((DMatrixRMaj) m.getMatrix()).data;
		for (int i = 0; i < data.length; i++)
			data[i] = f.apply(data[i]);
	}
	
	public static double sigmoid(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}
	
	/**
	 * Square of the Euclidian distance.
	 * Flattens matrices and treats them as N by M dimensional vectors.
	 * @param m1
	 * @param m2
	 * @return |v2 - v1|**2
	 */
	public static double distanceSq(SimpleMatrix v1, SimpleMatrix v2) {
		return v2.minus(v1).elementPower(2).elementSum();
	}
	
	/*
	 * TODO: Mean Squared Error, or Quadratic Cost Function
	 */
}
