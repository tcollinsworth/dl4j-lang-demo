package com.daisyworks.demo;

import io.vertx.core.Vertx;
import io.vertx.core.http.HttpMethod;
import io.vertx.ext.web.Router;
import io.vertx.ext.web.handler.BodyHandler;
import io.vertx.ext.web.handler.StaticHandler;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import com.daisyworks.demo.model.Evaluator;
import com.daisyworks.demo.model.Inferrer;
import com.daisyworks.demo.model.RecurrentNeuralNet;
import com.daisyworks.demo.model.Trainer;

//import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
//import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

/**
 * @author troy
 */
public class Service {
	static final int PORT = 8080;

	public static final int iterations = 100;
	public static final double learningRate = 0.02;
	public static final double regularizationL2 = 0.00001;

	public static int inputFeatureCnt = 226; // TODO from /src/main/resources/examples/charMap.txt
	public static int outputClassificationCnt = 7; // TODO from /src/main/resources/examples/classificationMap.txt

	public static int seed = 123;

	public static Map<Integer, String> classificationIdMap = new HashMap<Integer, String>();
	public static Map<String, Integer> classificationNameMap = new HashMap<String, Integer>();

	public static Map<Integer, String> charIdMap = new HashMap<Integer, String>();
	public static Map<String, Integer> charValMap = new HashMap<String, Integer>();

	public RecurrentNeuralNet rnn = new RecurrentNeuralNet(iterations, learningRate, inputFeatureCnt, outputClassificationCnt, seed, regularizationL2);

	// // infers or predicts classification for input observation features
	public Inferrer inferrer = new Inferrer(rnn);
	// // trains/fits a neural network model based on input observations and supervised labels
	public Trainer trainer = new Trainer(rnn);
	// // evaluates the precision and accuracy of a trained model for test/validation data
	public Evaluator evaluator = new Evaluator(rnn);

	public static void main(String[] args) throws IOException, InterruptedException {
		// for development, also requires staticHandler.setCacheEntryTimeout(1) and browser cache disable
		System.setProperty("vertx.disableFileCaching", "true");

		Service service = new Service();
		service.loadCharMap();
		service.loadClassificationMap();
		service.loadDataSets();
		// service.trainer.fit();

		Vertx vertx = Vertx.vertx();
		Router router = Router.router(vertx);
		router.route().handler(BodyHandler.create());
		// router.route(HttpMethod.POST, "/color-train-validate").blockingHandler(routingContext -> new
		// ColorRequestHandler(routingContext, service));
		router.route(HttpMethod.POST, "/modelAdmin").blockingHandler(routingContext -> new ModelAdminRequestHandler(routingContext, service));
		router.route("/*").handler(StaticHandler.create().setCacheEntryTimeout(1));

		vertx.createHttpServer().requestHandler(router::accept).listen(PORT, res -> {
			if (res.succeeded()) {
				System.out.println("Listening: " + PORT);
			} else {
				System.out.println("Failed to launch server: " + res.cause());
				System.exit(-1);
			}
		});
	}

	private static final int IDX = 0;
	private static final int VAL = 1;

	private void loadClassificationMap() throws IOException {
		String s = new String(Files.readAllBytes(new File("src/main/resources/examples/classificationMap.txt").toPath()));
		Arrays.asList(s.split("\n")).forEach((l) -> {
			String[] parts = l.split(":");

			classificationIdMap.put(Integer.parseInt(parts[IDX]), parts[VAL]);
			classificationNameMap.put(parts[VAL], Integer.parseInt(parts[IDX]));
		});
	}

	private void loadCharMap() throws IOException {
		String s = new String(Files.readAllBytes(new File("src/main/resources/examples/charMap.txt").toPath()));
		Arrays.asList(s.split("\n")).forEach((l) -> {
			String[] parts = l.split(":");

			charIdMap.put(Integer.parseInt(parts[IDX]), parts[VAL]);
			charValMap.put(parts[VAL], Integer.parseInt(parts[IDX]));
		});
	}

	public void loadDataSets() {
		// DefaultTokenizerFactory
	}
}
