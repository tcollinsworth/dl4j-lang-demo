package com.daisyworks.xor;

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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class XOR {
    public static void main(String[] args) {
        System.out.println("Starting...");

        DenseLayer inputLayer = new DenseLayer.Builder()
                .nIn(2)
                .nOut(3)
                .name("Input")
                .build();

        DenseLayer hiddenLayer = new DenseLayer.Builder()
                .nIn(3)
                .nOut(2)
                .name("Hidden")
                .build();

        OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(2)
                .nOut(2)
                .name("Output")
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new UniformDistribution(0, 1))
                .build();

        NeuralNetConfiguration.Builder nncBuilder = new NeuralNetConfiguration.Builder();
        nncBuilder.iterations(10000)
                .learningRate(0.1)
                .seed(123)
                .useDropConnect(false)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .biasInit(0)
                .miniBatch(false)
        ;

        NeuralNetConfiguration.ListBuilder listBuilder = nncBuilder.list();
        listBuilder.pretrain(false)
                .backprop(true);
        listBuilder.layer(0, inputLayer);
        listBuilder.layer(1, hiddenLayer);
        listBuilder.layer(2, outputLayer);

        listBuilder.backprop(true);

        MultiLayerNetwork myNetwork = new MultiLayerNetwork(listBuilder.build());
        myNetwork.init();


        System.out.println("Creating training data");

        //TODO import from data file with word2vec
        final int NUM_SAMPLES = 4;

        INDArray trainingInputs = Nd4j.zeros(NUM_SAMPLES, inputLayer.getNIn());
        INDArray trainingOutputs = Nd4j.zeros(NUM_SAMPLES, outputLayer.getNOut());

        // If 0,0 show 0
        trainingInputs.putScalar(new int[]{0,0}, 0);
        trainingInputs.putScalar(new int[]{0,1}, 0);
        trainingOutputs.putScalar(new int[]{0,0}, 1);
        trainingOutputs.putScalar(new int[]{0,1}, 0);

        // If 0,1 show 1
        trainingInputs.putScalar(new int[]{1,0}, 0);
        trainingInputs.putScalar(new int[]{1,1}, 1);
        trainingOutputs.putScalar(new int[]{1,0}, 0);
        trainingOutputs.putScalar(new int[]{1,1}, 1);

        // If 1,0 show 1
        trainingInputs.putScalar(new int[]{2,0}, 1);
        trainingInputs.putScalar(new int[]{2,1}, 0);
        trainingOutputs.putScalar(new int[]{2,0}, 0);
        trainingOutputs.putScalar(new int[]{2,1}, 1);

        // If 1,1 show 0
        trainingInputs.putScalar(new int[]{3,0}, 1);
        trainingInputs.putScalar(new int[]{3,1}, 1);
        trainingOutputs.putScalar(new int[]{3,0}, 1);
        trainingOutputs.putScalar(new int[]{3,1}, 0);

        DataSet myData = new DataSet(trainingInputs, trainingOutputs);

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

        myNetwork.fit(myData);

        // create output for every training sample
        INDArray output = myNetwork.output(myData.getFeatureMatrix());
        System.out.println(output);

        // let Evaluation prints stats how often the right output had the
        // highest value
//        Evaluation eval = new Evaluation(1);
//        eval.eval(myNetwork.getLabels(), output);
//        System.out.println(eval.stats());

        //TODO read from file and validate
        System.out.println("Predict/Infer");

        int[] results = myNetwork.predict(trainingInputs);
        System.out.println(Arrays.toString(results));

        INDArray testInputs = Nd4j.zeros(1, inputLayer.getNIn());
        testInputs.putScalar(new int[]{0,0}, 0);
        testInputs.putScalar(new int[]{0,1}, 0);
        int[] results2 = myNetwork.predict(testInputs);
        System.out.println(testInputs);
        System.out.println(Arrays.toString(results2));
        System.out.println(Arrays.toString(results));


        System.out.println("Completed");
    }
}
