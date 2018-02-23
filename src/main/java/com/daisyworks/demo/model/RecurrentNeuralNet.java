package com.daisyworks.demo.model;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author troy
 */
public class RecurrentNeuralNet {
	private final int inputFeatureCnt;
	private final int outputClassificationCnt;

	private int iterations;
	private double learningRate;
	private int seed; // initialization seed, keep same for reproducibility
	private double regularizationL2;

	MultiLayerNetwork net;

	public RecurrentNeuralNet(int iterations, double learningRate, int inputFeatureCnt, int outputClassificationCnt, int seed, double regularizationL2) {
		this.inputFeatureCnt = inputFeatureCnt;
		this.outputClassificationCnt = outputClassificationCnt;

		this.iterations = iterations;
		this.learningRate = learningRate;
		this.seed = seed;
		this.regularizationL2 = regularizationL2;

		initializeNewModel();
	}

	/**
	 * @param iterations
	 * @param learningRate
	 * @param seed
	 */
	public void initializeNewModel(int iterations, float learningRate, int seed) {
		this.iterations = iterations;
		this.learningRate = learningRate;
		this.seed = seed;

		initializeNewModel();
	}

	/**
	 * Create a brand new model.
	 */
	public void initializeNewModel() {
		NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder() //
				.iterations(iterations) //
				.learningRate(learningRate) //
				.seed(seed) //
				.updater(Updater.ADAM) //
				.regularization(true) //
				.l2(regularizationL2) //
				.weightInit(WeightInit.XAVIER) //
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue) //
				.gradientNormalizationThreshold(1.0) //
				.trainingWorkspaceMode(WorkspaceMode.SINGLE) //
				.inferenceWorkspaceMode(WorkspaceMode.SINGLE) //
				.list() //

				.pretrain(false) //
				.backprop(true) //

				.layer(0, new GravesLSTM.Builder() //
						.nIn(inputFeatureCnt) //
						.nOut(256) //
						.name("Input") //
						.activation(Activation.TANH) //
						.build()) //

				.layer(1, new RnnOutputLayer.Builder() //
						.nIn(256) //
						.nOut(outputClassificationCnt) //
						.name("Output") //
						.lossFunction(LossFunctions.LossFunction.MCXENT) //
						.activation(Activation.SOFTMAX) //
						.weightInit(WeightInit.DISTRIBUTION) //
						.dist(new UniformDistribution(0, 1)) //
						.build()); //

		MultiLayerNetwork net = new MultiLayerNetwork(listBuilder.build());
		net.init();
		this.net = net;

		// Print the number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for (int i = 0; i < layers.length; i++) {
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);

		net.setListeners(new ScoreIterationListener(100));
	}

	/**
	 * @param filePathName
	 *            i.e., trained_mnist_model.zip
	 * @param saveUpdater
	 *            allows additional training
	 * @throws IOException
	 */
	public void saveModel(String filePathName, boolean saveUpdater) throws IOException {
		File locationToSave = new File(filePathName);
		ModelSerializer.writeModel(net, locationToSave, saveUpdater);
	}

	/**
	 * @param filePathName
	 *            i.e., trained_mnist_model.zip
	 * @param saveUpdater
	 *            allows additional training
	 * @throws IOException
	 */
	public void restoreModel(String filePathName, boolean loadUpdater) throws IOException {
		File locationToSave = new File(filePathName);
		net = ModelSerializer.restoreMultiLayerNetwork(locationToSave, loadUpdater);
	}

}
