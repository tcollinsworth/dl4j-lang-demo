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

	private final int batchSize;

	private int numExamples = -1;

	private int cursor = 0;

	/**
	 * @param dir
	 * @param exampleLength
	 *            all examples will be truncated or padded to exampleLength
	 * @param inputCharSet
	 * @param classificationSet
	 */
	public ParagraphFileExampleIterator(String dir, int exampleLength, Map<Character, Integer> charValMap, String[] classificationSet, int batchSize) {
		this.dir = dir;
		this.exampleLength = exampleLength;

		this.charValMap = charValMap;
		this.classificationSet = classificationSet;

		this.batchSize = batchSize;
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
			int i = 0;
			Iterator<File> it = files.iterator();
			while (it.hasNext() && i++ < num) {
				File f = it.next();
				String example = new String(Files.readAllBytes(f.toPath()));
				if (example.length() > exampleLength) {
					example = example.substring(0, exampleLength); // ensure max length
				}
				examples.add(example);
				consumedExamples.add(f);
			}

			INDArray inputFeatureMatrix = Nd4j.create(new int[] { examples.size(), charValMap.size(), exampleLength }, 'f');
			INDArray labels = Nd4j.create(new int[] { examples.size(), classificationSet.length, exampleLength }, 'f');
			// Masks 1 if data present, 0 for padding
			INDArray featuresMask = Nd4j.zeros(examples.size(), exampleLength);
			INDArray labelsMask = Nd4j.zeros(examples.size(), exampleLength);

			for (i = 0; i < examples.size(); i++) {
				String example = examples.get(i);

				INDArrayIndex[] indices = new INDArrayIndex[] { //
				NDArrayIndex.point(i), //
						NDArrayIndex.all(), //
						NDArrayIndex.interval(0, exampleLength) };
				inputFeatureMatrix.put(indices, getExampleMatrix(example));

				// TODO data mask vector
				// TODO label
				// TODO label mask
			}

			return new DataSet(inputFeatureMatrix, labels, featuresMask, labelsMask);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	private INDArray getExampleMatrix(String example) {
		INDArray exampleMatrix = Nd4j.zeros(exampleLength, charValMap.size());
		for (int i = 0; i < example.length(); i++) {
			Integer charIdx = charValMap.get(example.charAt(i));
			// if not in map for some reason, leave as zero
			if (charIdx == null) {
				continue;
			}
			// 1-hot encode char
			exampleMatrix.putScalar(new int[] { i, charIdx.intValue() }, 1);
		}
		return exampleMatrix;
	}

	@Override
	public boolean hasNext() {
		return cursor < numExamples();
	}

	@Override
	public DataSet next() {
		return next(batchSize);
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
		return batchSize;
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
