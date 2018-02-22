package com.daisyworks.demo.model;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author troy
 *
 */
public class NeuralNet {
	private final int inputFeatureCnt;
	private final int outputClassificationCnt;

	private int iterations;
	private float learningRate;

	MultiLayerNetwork net;

	public NeuralNet(int iterations, float learningRate, int inputFeatureCnt, int outputClassificationCnt) {
		this.inputFeatureCnt = inputFeatureCnt;
		this.outputClassificationCnt = outputClassificationCnt;

		this.iterations = iterations;
		this.learningRate = learningRate;

		initializeNewModel();
	}

	/**
	 * @param iterations
	 * @param learningRate
	 */
	public void initializeNewModel(int iterations, float learningRate) {
		this.iterations = iterations;
		this.learningRate = learningRate;

		initializeNewModel();
	}

	/**
	 * Create a brand new model.
	 */
	public void initializeNewModel() {
		NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder() //
				.iterations(iterations) //
				.learningRate(learningRate) //
				.seed(123) //
				.useDropConnect(false) //
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) //
				.biasInit(0) //
				.miniBatch(false) //
				.list() //

				.pretrain(false) //
				.backprop(true) //

				.layer(0, new DenseLayer.Builder() //
						.nIn(inputFeatureCnt) //
						.nOut(inputFeatureCnt) //
						.name("Input") //
						.build()) //

				.layer(1, new DenseLayer.Builder() //
						.nIn(inputFeatureCnt) //
						.nOut(outputClassificationCnt) //
						.name("Hidden") //
						.build()) //

				.layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //
						.nIn(outputClassificationCnt) //
						.nOut(outputClassificationCnt) //
						.name("Output") //
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

	/**
	 * Not static class because the feature size is dependent on the neural net configuration.
	 *
	 */
	public class Observation {
		public final float[] features = new float[inputFeatureCnt];
		public final int classificationIdx;

		/**
		 * Only set inputs that are relevant. Others will be set to zero.
		 * 
		 * @param classificationIdx
		 * @param f
		 */
		public Observation(int classificationIdx, float... f) {
			this.classificationIdx = classificationIdx;
			for (int i = 0; i < f.length; i++) {
				features[i] = f[i];
			}
		}

		public Observation(float... f) {
			this.classificationIdx = -1;
			for (int i = 0; i < f.length; i++) {
				features[i] = f[i];
			}
		}

		public String toString() {
			return String.format("ClassificationIdx: %d\r\nFeatures: \r\n%s", classificationIdx, Arrays.toString(features));
		}
	}
}
