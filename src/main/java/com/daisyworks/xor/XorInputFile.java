package com.daisyworks.xor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class XorInputFile {
	public static void main(String[] args) throws IOException, InterruptedException {
		System.out.println("Starting...");

		// 10% learning rate takes 400 iterations to memorize JK FF
		// 10% learning rate takes 1500 iterations to memorize JK FF perfectly
		int interations = 4000;
		float learningRate = 0.1f;
		int inputCnt = 5;

		NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
				.iterations(interations)
				.learningRate(learningRate)
				.seed(123)
				.useDropConnect(false)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.biasInit(0)
				.miniBatch(false)
				.list()

				.pretrain(false)
				.backprop(true)

				.layer(0, new DenseLayer.Builder().nIn(inputCnt).nOut(inputCnt).name("Input").build())

				.layer(1, new DenseLayer.Builder().nIn(inputCnt).nOut(2).name("Hidden").build())

				.layer(2,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(2).nOut(2).name("Output").activation(Activation.SOFTMAX)
								.weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1)).build());

		MultiLayerNetwork myNetwork = new MultiLayerNetwork(listBuilder.build());
		myNetwork.init();

		System.out.println("Reading training data");

		int skipNumLines = 1;
		char delimiter = ',';

		RecordReader recordReader = new CSVRecordReader(skipNumLines, delimiter);
		recordReader.initialize(new FileSplit(new ClassPathResource("jk-ff-data.csv").getFile()));

		int labelIndex = inputCnt; // 7 data bits followed by 1 label
		int numClasses = 2; // 2 classes in the data set. Classes have integer values 0 = false, 1 = true
		int batchSize = 30;

		DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
		DataSet allData = iterator.next();

		List<String> labelNames = new ArrayList<>();
		labelNames.add("F");
		labelNames.add("T");
		allData.setLabelNames(labelNames);

		System.out.println(allData);

		System.out.println("Fitting");

		// Print the number of parameters in the network (and for each layer)
		Layer[] layers = myNetwork.getLayers();
		int totalNumParams = 0;
		for (int i = 0; i < layers.length; i++) {
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);

		myNetwork.setListeners(new ScoreIterationListener(100));

		myNetwork.fit(allData);

		// create output for every training sample
		INDArray output = myNetwork.output(allData.getFeatureMatrix());
		System.out.println(output);

		// let Evaluation prints stats how often the right output had the
		// highest value
		Evaluation eval = new Evaluation(1);
		eval.eval(myNetwork.getLabels(), output);
		System.out.println(eval.stats());

		System.out.println("Predict/Infer");

		// read from file and validate
		System.out.println(allData);
		List<String> results = myNetwork.predict(allData);
		System.out.println(results);

		System.out.println("Timing");
		INDArray testInputs = Nd4j.zeros(1, 5);
		testInputs.putScalar(new int[] { 0, 0 }, 0.3);
		testInputs.putScalar(new int[] { 0, 1 }, 0);
		testInputs.putScalar(new int[] { 0, 2 }, 1);
		testInputs.putScalar(new int[] { 0, 3 }, 0);
		testInputs.putScalar(new int[] { 0, 4 }, 0);

		long start = System.nanoTime();
		int[] results2 = myNetwork.predict(testInputs); // 512us for 1 row
		long time = System.nanoTime() - start;

		System.out.println(time + " ns");
		System.out.println("results2: " + Arrays.toString(results2));

		System.out.println("Completed");
	}
}
