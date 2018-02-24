package com.daisyworks.demo;

import io.vertx.core.Vertx;
import io.vertx.core.http.HttpMethod;
import io.vertx.ext.web.Router;
import io.vertx.ext.web.handler.BodyHandler;
import io.vertx.ext.web.handler.StaticHandler;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.daisyworks.demo.model.Evaluator;
import com.daisyworks.demo.model.Inferrer;
import com.daisyworks.demo.model.RecurrentNeuralNet;
import com.daisyworks.demo.model.Trainer;
import com.daisyworks.language.ParagraphFileExampleIterator;

/**
 * @author troy
 */
public class Service {
	private static final int IDX = 0;
	private static final int VAL = 1;

	private final int PORT = 8080;

	private final int iterations = 100;
	private final double learningRate = 0.02;
	private final double regularizationL2 = 0.00001;

	private int inputFeatureCnt; // characters
	private int outputClassificationCnt; // classifications

	private int exampleLength = 422; // TODO read from longest in dir discovered when parsing

	private int seed = 123;

	private String[] classificationSet;
	private Map<String, Integer> classificationNameMap = new HashMap<String, Integer>();

	private Character[] characterSet;
	private Map<Character, Integer> charValMap = new HashMap<Character, Integer>();

	private ParagraphFileExampleIterator trainDataSetIterator;
	private ParagraphFileExampleIterator validationDataSetIterator;
	private ParagraphFileExampleIterator testDataSetIterator;

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

		s.loadInputCharacterSet();
		s.loadOutputClassificationSet();

		s.rnn = new RecurrentNeuralNet(s.iterations, s.learningRate, s.inputFeatureCnt, s.outputClassificationCnt, s.seed, s.regularizationL2);

		s.inferrer = new Inferrer(s.rnn);
		s.trainer = new Trainer(s.rnn);
		s.evaluator = new Evaluator(s.rnn);

		s.loadDataSets();

		s.trainer.fit(s.trainDataSetIterator);

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

	private void loadOutputClassificationSet() throws IOException {
		List<String> classes = new ArrayList<>();

		String s = new String(Files.readAllBytes(new File("src/main/resources/examples/classificationMap.txt").toPath()));
		Arrays.asList(s.split("\n")).forEach((l) -> {
			String[] parts = l.split(":");

			classes.add(parts[VAL]);
			classificationNameMap.put(parts[VAL], Integer.parseInt(parts[IDX]));
		});
		classificationSet = classes.toArray(new String[0]);
		outputClassificationCnt = classes.size();

	}

	private void loadInputCharacterSet() throws IOException {
		List<Character> chars = new ArrayList<>();

		String s = new String(Files.readAllBytes(new File("src/main/resources/examples/charMap.txt").toPath()));
		Arrays.asList(s.split("\n")).forEach((l) -> {
			String[] parts = l.split(":");
			chars.add((char) Integer.parseInt(parts[VAL]));
			charValMap.put((char) Integer.parseInt(parts[VAL]), Integer.parseInt(parts[IDX]));
		});
		characterSet = chars.toArray(new Character[0]);
		inputFeatureCnt = chars.size();
	}

	private void loadDataSets() {
		trainDataSetIterator = new ParagraphFileExampleIterator("src/main/resources/examples/train", exampleLength, charValMap, classificationSet, -1);

		validationDataSetIterator = new ParagraphFileExampleIterator("src/main/resources/examples/validation", exampleLength, charValMap, classificationSet, -1);

		testDataSetIterator = new ParagraphFileExampleIterator("src/main/resources/examples/test", exampleLength, charValMap, classificationSet, -1);

		trainDataSetIterator.next(100);
		validationDataSetIterator.next(100);
		testDataSetIterator.next(100);
	}
}
