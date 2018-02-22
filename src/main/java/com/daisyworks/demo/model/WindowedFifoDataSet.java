package com.daisyworks.demo.model;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.daisyworks.demo.model.NeuralNet.Observation;

/**
 * Not a real Nd4j DataSet, contains rolling window INDArrays of features and labels
 * 
 * @author troy
 *
 */
public class WindowedFifoDataSet {
	private final String purpose;
	private final int observationWindowSize;
	private final int inputFeatureCnt;
	// private final int outputClassificationCnt;
	private int observationIdx = 0;
	public INDArray features;
	public INDArray classifications;

	/**
	 * @param window
	 *            = rows
	 * @param inputs
	 */
	public WindowedFifoDataSet(String purpose, int observationWindowSize, int inputFeatureCnt, int outputClassificationCnt) {
		this.purpose = purpose;
		this.observationWindowSize = observationWindowSize;
		this.inputFeatureCnt = inputFeatureCnt;
		// this.outputClassificationCnt = outputClassificationCnt;

		features = Nd4j.zeros(observationWindowSize, inputFeatureCnt);
		classifications = Nd4j.zeros(observationWindowSize, outputClassificationCnt);
	}

	public void addObservation(Observation f) {
		addFeature(f);

		// if first observation, fill array - ideally grow the array as observations occur, that's more difficult
		while (observationIdx < observationWindowSize) {
			addFeature(f);
		}
		// System.out.println("features: " + features);
		// System.out.println("labels: " + classifications);
	}

	private void addFeature(Observation f) {
		// System.out.println(String.format("purpose: %s, observationIdx: %d\r\nobservation: %s", purpose,
		// observationIdx % observationWindowSize, f));
		for (int fIdx = 0; fIdx < inputFeatureCnt; fIdx++) {
			features.putScalar(new int[] { observationIdx % observationWindowSize, fIdx }, f.features[fIdx]);
			addLabels(f.classificationIdx);
		}
		++observationIdx;
	}

	private void addLabels(int clazzIdx) {
		for (int colIdx = 0; colIdx < classifications.columns(); colIdx++) {
			classifications.putScalar(new int[] { observationIdx % observationWindowSize, colIdx }, colIdx == clazzIdx ? 1f : 0f);
		}
	}

	public String toString() {
		return String.format("purpose %s, row: %d, features: %s, labels: %s", purpose, observationIdx, features, classifications);
	}
}
