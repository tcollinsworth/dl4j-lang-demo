package com.daisyworks.demo.model;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import com.daisyworks.demo.Service;

/**
 * @author troy
 *
 */
public class Trainer {
	RecurrentNeuralNet rnn;

	public Trainer(RecurrentNeuralNet rnn) {
		this.rnn = rnn;

		// https://deeplearning4j.org/workspaces
		// limit gc frequency - 10000 milliseconds
		// Nd4j.getMemoryManager().setAutoGcWindow(10000);

		// OR disable
		// Nd4j.getMemoryManager().togglePeriodicGc(false);
	}

	public void train(Service service) {
		// rnn.net.fit(service.trainColoData.features, service.trainColoData.classifications);
	}

	int fitCnt = 0;

	public void fit(DataSetIterator trainDataSetIterator) {
		rnn.net.fit(trainDataSetIterator);
	}
}
