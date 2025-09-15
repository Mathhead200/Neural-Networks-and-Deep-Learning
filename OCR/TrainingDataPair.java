import org.ejml.simple.SimpleMatrix;

public interface TrainingDataPair {
	SimpleMatrix input() throws Exception;
	SimpleMatrix expectedOutput();
}
