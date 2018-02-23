package com.daisyworks.demo.model;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

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
		Nd4j.getMemoryManager().setAutoGcWindow(10000);

		// OR disable
		// Nd4j.getMemoryManager().togglePeriodicGc(false);
	}

	public void train(Service service) {
		// nn.net.fit(service.trainColoData.features, service.trainColoData.classifications);
	}

	public void fit() throws IOException, InterruptedException {
		System.out.println("Reading training data");

		int skipNumLines = 1;
		char delimiter = ',';

		RecordReader recordReader = new CSVRecordReader(skipNumLines, delimiter);
		recordReader.initialize(new FileSplit(new ClassPathResource("color-data.csv").getFile()));

		int labelIndex = 5;
		int numClasses = 11;
		int batchSize = 30;

		DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
		DataSet allData = iterator.next();

		List<String> labelNames = new ArrayList<>();
		labelNames.add("white");
		labelNames.add("grey");
		labelNames.add("black");
		labelNames.add("brown");
		labelNames.add("red");
		labelNames.add("orange");
		labelNames.add("yellow");
		labelNames.add("green");
		labelNames.add("blue");
		labelNames.add("violet");
		labelNames.add("pink");
		allData.setLabelNames(labelNames);

		System.out.println(allData);

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

		rnn.net.fit(allData);

		// create output for every training sample - test or validation data
		INDArray output = rnn.net.output(allData.getFeatureMatrix());
		System.out.println(output);

		// let Evaluation prints stats how often the right output had the
		// highest value
		Evaluation eval = new Evaluation(11);
		eval.eval(rnn.net.getLabels(), output);
		System.out.println(eval.stats());
	}
}
