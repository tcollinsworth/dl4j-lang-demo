package com.daisyworks.language;

import java.util.List;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class LanguageExampleIterator implements DataSetIterator {
	private static final long serialVersionUID = 1L;

	@Override
	public boolean hasNext() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public DataSet next() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public DataSet next(int num) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int totalExamples() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int inputColumns() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int totalOutcomes() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public boolean resetSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean asyncSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void reset() {
		// TODO Auto-generated method stub

	}

	@Override
	public int batch() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int cursor() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int numExamples() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		// TODO Auto-generated method stub

	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<String> getLabels() {
		// TODO Auto-generated method stub
		return null;
	}
}
