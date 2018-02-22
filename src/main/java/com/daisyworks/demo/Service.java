package com.daisyworks.demo;

import io.vertx.core.Vertx;
import io.vertx.ext.web.Router;
import io.vertx.ext.web.handler.BodyHandler;
import io.vertx.ext.web.handler.StaticHandler;

import java.io.IOException;

/**
 * @author troy
 *
 */
public class Service {
	static final int PORT = 8080;

	public static final int iterations = 100;
	public static final float learningRate = 0.01f;

	// public NeuralNet nn = new NeuralNet(iterations, learningRate, inputFeatureCnt, outputClassificationCnt);
	// // infers or predicts classification for input observation features
	// public Inferrer inferrer = new Inferrer(nn);
	// // trains/fits a neural network model based on input observations and supervised labels
	// public Trainer trainer = new Trainer(nn);
	// // evaluates the precision and accuracy of a trained model for test/validation data
	// public Evaluator evaluator = new Evaluator(nn);

	public static void main(String[] args) throws IOException, InterruptedException {
		// for development, also requires staticHandler.setCacheEntryTimeout(1) and browser cache disable
		System.setProperty("vertx.disableFileCaching", "true");

		Service service = new Service();
		service.createNewDataSets();
		// service.trainer.fit();

		Vertx vertx = Vertx.vertx();
		Router router = Router.router(vertx);
		router.route().handler(BodyHandler.create());
		// router.route(HttpMethod.POST, "/color-train-validate").blockingHandler(routingContext -> new
		// ColorRequestHandler(routingContext, service));
		// router.route(HttpMethod.POST, "/modelAdmin").blockingHandler(routingContext -> new
		// ModelAdminRequestHandler(routingContext, service));
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

	public void createNewDataSets() {

	}
}
