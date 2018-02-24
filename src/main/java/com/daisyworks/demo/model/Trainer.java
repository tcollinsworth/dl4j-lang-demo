package com.daisyworks.demo.model;

import java.io.IOException;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.factory.Nd4j;

import com.daisyworks.demo.Service;
import com.daisyworks.language.ParagraphFileExampleIterator;

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
		Nd4j.getMemoryManager().setAutoGcWindow(10000);

		// OR disable
		// Nd4j.getMemoryManager().togglePeriodicGc(false);
	}

	public void train(Service service) {
		// rnn.net.fit(service.trainColoData.features, service.trainColoData.classifications);
	}

	public void fit(ParagraphFileExampleIterator trainDataSetIterator) throws IOException, InterruptedException {
		System.out.println("Fitting");

		// Print the number of parameters in the network (and for each layer)
		Layer[] layers = rnn.net.getLayers();
		int totalNumParams = 0;
		for (int i = 0; i < layers.length; i++) {
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);

		rnn.net.setListeners(new ScoreIterationListener(100));

		rnn.net.fit(trainDataSetIterator);

		// // create output for every training sample - test or validation data
		// INDArray output = rnn.net.output(trainDataSetIterator.getFeatureMatrix());
		// System.out.println(output);
		//
		// // let Evaluation print stats how often the right output had the
		// // highest value
		// Evaluation eval = new Evaluation(11);
		// eval.eval(rnn.net.getLabels(), output);
		// System.out.println(eval.stats());
	}
}
