package com.daisyworks.demo.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.daisyworks.demo.Service;
import com.daisyworks.language.TokenizeSentenceIntoWords;

/**
 * @author troy
 *
 */
public class Inferrer {
	private final Service service;
	private final RecurrentNeuralNet rnn;
	private final Map<Character, Double> charMap;

	public Inferrer(RecurrentNeuralNet rnn, Map<Character, Double> charMap, Service service) {
		this.service = service;
		this.rnn = rnn;
		this.charMap = charMap;
	}

	public Output infer(String rawExample) {
		// pre-process remove numbers and punctuation, toLowerCase, trim
		List<String> words = TokenizeSentenceIntoWords.tokenize(rawExample);

		final StringBuilder sb = new StringBuilder();
		String[] delimiter = { "" };
		words.forEach((word) -> {
			sb.append(delimiter[0]).append(word);
			delimiter[0] = " ";
		});

		String example = sb.toString().trim().toLowerCase();

		// encode/scale, remove untrained characters,
		List<Double> charsScaledEncoded = new ArrayList<>();
		example.chars().forEachOrdered((c) -> {
			Double charScaledEncoding = charMap.get((char) c);
			if (charScaledEncoding != null) {
				charsScaledEncoded.add(charScaledEncoding);
			}
		});

		// vectorize inputs, create input mask
		// input Dimensions [miniBatchSize,inputSize,inputTimeSeriesLength]
		INDArray input = Nd4j.zeros(new int[] { 1, 1, charsScaledEncoded.size() }, 'f');
		for (int i = 0; i < charsScaledEncoded.size(); i++) {
			input.putScalar(new int[] { 0, 0, i }, charsScaledEncoded.get(i));
		}

		// System.out.println("infer inputs: " + inputs.toString());

		long start = System.nanoTime();

		rnn.net.rnnClearPreviousState();
		// Output dimensions [miniBatchSize,outputSize] or 1 x 7 languages
		INDArray outputs = rnn.net.rnnTimeStep(input);
		// System.out.println(outputs);

		// System.out.println(outputs.length);
		// System.out.println(Arrays.toString(outputs));
		// System.out.println(nn.net.summary());
		long timeNs = System.nanoTime() - start;
		// System.out.println(timeNs);
		float timeMs = ((float) timeNs) / 1000000;

		return getOutput(outputs, charsScaledEncoded.size(), timeMs);
	}

	private Output getOutput(INDArray outputs, int inputTimeSeriesLength, float timeMs) {
		return new Output(timeMs, outputs);
	}

	public class Output {
		public int classificationIdx = 0;
		public final List<String> classificationProbabilities = new ArrayList<>();
		public final float timeMs;
		public final String probMatrix;

		public Output(float timeMs, INDArray outputMatrix) {
			this.timeMs = timeMs;
			this.probMatrix = outputMatrix.toString();

			//System.out.println(outputMatrix.shapeInfoToString());

			INDArrayIndex[] probIndices = new INDArrayIndex[] { //
					NDArrayIndex.point(0), //
					NDArrayIndex.all(), //
					NDArrayIndex.all(), //
			};
			INDArray probMatrix = outputMatrix.get(probIndices);

			//Give earlier char probabilities lower weight, later char probabilities more weight
			int multiplierCnt = (probMatrix.columns() * (probMatrix.columns() + 1)) / 2; // * charSeqIdx+1 | * 1,2,3,n+1
			float[] classProbs = new float[probMatrix.rows()];
			for (int classIdx = 0; classIdx < probMatrix.rows(); classIdx++) {
				for (int charSeqIdx = 0; charSeqIdx < probMatrix.columns(); charSeqIdx++) {
					classProbs[classIdx] += (probMatrix.getFloat(classIdx, charSeqIdx) * (charSeqIdx+1)); 
				}
			}

			float maxClassProb = 0;
			for (int classIdx = 0; classIdx < probMatrix.rows(); classIdx++) {
				classProbs[classIdx] = classProbs[classIdx] / multiplierCnt; //probMatrix.columns();
				classificationProbabilities.add(service.getClassifications()[classIdx] + ":" + classProbs[classIdx]);
				if (classProbs[classIdx] > maxClassProb) {
					maxClassProb = classProbs[classIdx];
					classificationIdx = classIdx;
				}
			}
		}
	}
}
