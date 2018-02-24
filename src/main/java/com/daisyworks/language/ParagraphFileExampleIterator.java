package com.daisyworks.language;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class ParagraphFileExampleIterator implements DataSetIterator {
	private static final long serialVersionUID = 1L;

	private final Map<Character, Integer> charValMap;
	private final String[] classificationSet;

	private final String dir;
	private Set<File> consumedExamples = new HashSet<File>();

	private final int exampleLength; // length to truncate or pad examples to

	private final int miniBatchSize;

	private int numExamples = -1; // set from example file count in train/validation/test dir

	private int cursor = 0;

	/**
	 * @param dir
	 * @param exampleLength
	 *            all examples will be truncated or padded to exampleLength
	 * @param inputCharSet
	 * @param classificationSet
	 */
	public ParagraphFileExampleIterator(String dir, int exampleLength, Map<Character, Integer> charValMap, String[] classificationSet, int miniBatchSize) {
		this.dir = dir;
		this.exampleLength = exampleLength;

		this.charValMap = charValMap;
		this.classificationSet = classificationSet;

		this.miniBatchSize = miniBatchSize;
	}

	/**
	 * @param num
	 *            mini-batch size
	 * @return
	 */
	@Override
	public DataSet next(int num) {
		List<String> examples = new ArrayList<>();
		try {
			// get file list of all examples/observation files in dir
			List<File> files = new ArrayList<>(Arrays.asList(new File(dir).listFiles()));
			// remove what's already been consumed by iterator
			files.removeAll(consumedExamples);
			// randomize the examples/observations
			Collections.shuffle(files);

			// load each observation
			int exampleIdx = 0;
			Iterator<File> fileIterator = files.iterator();
			while (fileIterator.hasNext() && exampleIdx++ < num) {
				File f = fileIterator.next();
				String example = new String(Files.readAllBytes(f.toPath()));
				if (example.length() > exampleLength) {
					example = example.substring(0, exampleLength); // ensure max length
				}
				examples.add(example);
				consumedExamples.add(f);
				cursor++;
			}

			// Matrix for all examples in miniBatch
			INDArray inputFeatureMatrix = Nd4j.create(new int[] { examples.size(), charValMap.size(), exampleLength }, 'f');
			INDArray labelsMatrix = Nd4j.create(new int[] { examples.size(), classificationSet.length, exampleLength }, 'f');
			// Masks 1 if data present, 0 for padding
			INDArray featuresMaskMatrix = Nd4j.zeros(examples.size(), exampleLength);
			INDArray labelsMaskMatrix = Nd4j.zeros(examples.size(), exampleLength);

			for (exampleIdx = 0; exampleIdx < examples.size(); exampleIdx++) {
				String example = examples.get(exampleIdx);

				INDArrayIndex[] indices = new INDArrayIndex[] { //
				NDArrayIndex.point(exampleIdx), //
						NDArrayIndex.all(), //
						NDArrayIndex.interval(0, example.length()) };
				inputFeatureMatrix.put(indices, getExampleMatrix(example));

				// for current example, set each corresponding feature mask value to 1 for the length of the example,
				// leaving padding values 0
				featuresMaskMatrix.get(new INDArrayIndex[] { NDArrayIndex.point(exampleIdx), NDArrayIndex.interval(0, example.length()) }).assign(1);

				int classIdx = getLabelClassIdx(files.get(exampleIdx).getName());
				int labelAtLastFeatureIdx = examples.get(exampleIdx).length() - 1;
				labelsMatrix.putScalar(new int[] { exampleIdx, classIdx, labelAtLastFeatureIdx }, 1.0);

				labelsMaskMatrix.putScalar(new int[] { exampleIdx, labelAtLastFeatureIdx }, 1.0);
			}

			return new DataSet(inputFeatureMatrix, labelsMatrix, featuresMaskMatrix, labelsMaskMatrix);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	private int getLabelClassIdx(String fileName) {
		for (int classIdx = 0; classIdx < classificationSet.length; classIdx++) {
			if (fileName.startsWith(classificationSet[classIdx])) {
				return classIdx;
			}
		}
		throw new RuntimeException("classification not found for " + fileName);
	}

	private INDArray getExampleMatrix(String example) {
		INDArray exampleMatrix = Nd4j.zeros(exampleLength, charValMap.size());
		System.out.println("***********************************");
		System.out.println(exampleMatrix.shapeInfoToString());
		System.out.println(example.length());
		for (int exampleCharIdx = 0; exampleCharIdx < example.length(); exampleCharIdx++) {
			Integer charMapIdx = charValMap.get(example.charAt(exampleCharIdx));
			// if not in map for some reason, leave as zero
			if (charMapIdx == null) {
				// System.out.println(String.format("null %s, %d, %d, %d", example.charAt(i), (int) example.charAt(i),
				// i, example.length()));
				throw new RuntimeException("unrecognized example charAt " + exampleCharIdx + " " + example);
				// continue;
			}
			// 1-hot encode char
			exampleMatrix.putScalar(new int[] { exampleCharIdx, charMapIdx.intValue() }, 1);
		}
		System.out.println(exampleMatrix.length());
		return exampleMatrix;
	}

	@Override
	public boolean hasNext() {
		return cursor < numExamples();
	}

	@Override
	public DataSet next() {
		return next(miniBatchSize);
	}

	@Override
	public int totalExamples() {
		if (numExamples != -1) {
			return numExamples;
		}
		return numExamples = new File(dir).listFiles().length;
	}

	@Override
	public int inputColumns() {
		return charValMap.size();
	}

	@Override
	public int totalOutcomes() {
		return classificationSet.length;
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public boolean asyncSupported() {
		return true;
	}

	@Override
	public void reset() {
		consumedExamples.clear();
		cursor = 0;
	}

	@Override
	public int batch() {
		return miniBatchSize;
	}

	@Override
	public int cursor() {
		return cursor;
	}

	@Override
	public int numExamples() {
		return totalExamples();
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		throw new UnsupportedOperationException();
	}

	@Override
	public List<String> getLabels() {
		return Arrays.asList(classificationSet);
	}
}
