package com.daisyworks.demo.model;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.daisyworks.demo.model.NeuralNet.Observation;

/**
 * @author troy
 *
 */
public class Inferrer {
	NeuralNet nn;

	public Inferrer(NeuralNet nn) {
		this.nn = nn;
	}

	public Output infer(Observation f) {
		INDArray inputs = Nd4j.zeros(1, 5);
		for (int i = 0; i < f.features.length; i++) {
			inputs.putScalar(new int[] { 0, i }, f.features[i]);
		}

		// System.out.println("infer inputs: " + inputs.toString());

		long start = System.nanoTime();

		INDArray classificationProbabilities = nn.net.output(inputs);
		// System.out.println(labelProbabilities);

		int[] outputs = nn.net.predict(inputs); // 512us for 1 row
		// System.out.println(outputs.length);
		// System.out.println(Arrays.toString(outputs));
		// System.out.println(nn.net.summary());
		long timeNs = System.nanoTime() - start;
		// System.out.println(timeNs);
		float timeMs = ((float) timeNs) / 1000000;

		return new Output(outputs, getList(classificationProbabilities), timeMs);
	}

	private List<Float> getList(INDArray classificationProbabilities) {
		List<Float> list = new ArrayList<>();
		for (int i = 0; i < classificationProbabilities.length(); i++) {
			list.add(classificationProbabilities.getFloat(i));
		}
		return list;
	}

	public static class Output {
		public final int[] outputs;
		public final List<Float> classificationProbabilities;
		public final float timeMs;

		public Output(int[] outputs, List<Float> classificationProbabilities, float timeMs) {
			this.outputs = outputs;
			this.classificationProbabilities = classificationProbabilities;
			this.timeMs = timeMs;
		}
	}
}
