package com.daisyworks.demo;

import io.vertx.core.Vertx;
import io.vertx.core.http.HttpMethod;
import io.vertx.ext.web.Router;
import io.vertx.ext.web.handler.BodyHandler;
import io.vertx.ext.web.handler.StaticHandler;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import com.daisyworks.demo.model.Evaluator;
import com.daisyworks.demo.model.Inferrer;
import com.daisyworks.demo.model.RecurrentNeuralNet;
import com.daisyworks.demo.model.Trainer;
import com.daisyworks.language.DataLoader;

/**
 * @author troy
 */
public class Service {
	private final int PORT = 8080;

	public int miniBatchSize = 10;
	private int seed = 123;
	private final int iterations = 1;
	private final double learningRate = 0.02;
	private final double regularizationL2 = 0.00001;

	public int inputFeatureCnt; // characters
	public int outputClassificationCnt; // classifications

	// The char length of longest example for truncating/padding
	public int maxExampleLength;

	public String[] classificationSet;
	public Map<String, Integer> classificationNameMap = new HashMap<String, Integer>();

	public Character[] characterSet;
	public Map<Character, Integer> charValMap = new HashMap<Character, Integer>();

	public DataSetIterator trainDataSetIterator;
	public DataSetIterator validationDataSetIterator;
	public DataSetIterator testDataSetIterator;

	private RecurrentNeuralNet rnn;

	// // infers or predicts classification for input observation features
	private Inferrer inferrer;
	// // trains/fits a neural network model based on input observations and supervised labels
	private Trainer trainer;
	// // evaluates the precision and accuracy of a trained model for test/validation data
	private Evaluator evaluator;

	public static void main(String[] args) throws IOException, InterruptedException {
		Service s = new Service();
		// for development, also requires staticHandler.setCacheEntryTimeout(1) and browser cache disable
		System.setProperty("vertx.disableFileCaching", "true");

		DataLoader dataLoader = new DataLoader(s);

		dataLoader.loadInputCharacterSet();
		dataLoader.loadOutputClassificationSet();
		dataLoader.loadDataSetStats();
		dataLoader.loadDataSets();

		s.rnn = new RecurrentNeuralNet(s.iterations, s.learningRate, s.inputFeatureCnt, s.outputClassificationCnt, s.seed, s.regularizationL2);

		s.inferrer = new Inferrer(s.rnn);
		s.trainer = new Trainer(s.rnn);
		s.evaluator = new Evaluator(s.rnn);

		for (int i = 0; i < 100; i++) {
			s.trainer.fit(s.trainDataSetIterator);
		}

		Vertx vertx = Vertx.vertx();
		Router router = Router.router(vertx);
		router.route().handler(BodyHandler.create());
		// router.route(HttpMethod.POST, "/color-train-validate").blockingHandler(routingContext -> new
		// ColorRequestHandler(routingContext, service));
		router.route(HttpMethod.POST, "/modelAdmin").blockingHandler(routingContext -> new ModelAdminRequestHandler(routingContext, s));
		router.route("/*").handler(StaticHandler.create().setCacheEntryTimeout(1));

		vertx.createHttpServer().requestHandler(router::accept).listen(s.PORT, res -> {
			if (res.succeeded()) {
				System.out.println("Listening: " + s.PORT);
			} else {
				System.out.println("Failed to launch server: " + res.cause());
				System.exit(-1);
			}
		});
	}
}
