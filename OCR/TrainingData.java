import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

@SuppressWarnings("serial")
public class TrainingData extends ArrayList<TrainingDataPair> {
	public void shuffle(Random rng) {
		Collections.shuffle(this, rng);
	}
}
