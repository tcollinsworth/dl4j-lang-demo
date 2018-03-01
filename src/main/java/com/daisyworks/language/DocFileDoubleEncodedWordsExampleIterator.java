package com.daisyworks.language;

import java.io.File;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class DocFileDoubleEncodedWordsExampleIterator implements DataSetIterator {
	private static final long serialVersionUID = 1L;

	private final String[] classificationSet;

	private final String dir;
	private Set<File> consumedExamples = new HashSet<File>();

	// The char length of longest example for truncating/padding
	private final int maxExampleLength;

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
	public DocFileDoubleEncodedWordsExampleIterator(String dir, int exampleLength, Map<Character, Integer> charValMap, String[] classificationSet, int miniBatchSize) {
		this.dir = dir;

		this.maxExampleLength = exampleLength;
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
		return null;
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
		return 1;
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
