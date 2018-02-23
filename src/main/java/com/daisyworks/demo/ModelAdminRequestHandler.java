package com.daisyworks.demo;

import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.RoutingContext;

public class ModelAdminRequestHandler extends RequestHandler {

	public ModelAdminRequestHandler(RoutingContext rc, Service service) {
		super(rc, service);
	}

	@Override
	public void handle() {
		boolean saveModel = bodyJson.getBoolean("saveModel", false);
		boolean resetModel = bodyJson.getBoolean("resetModel", false);
		boolean loadModel = bodyJson.getBoolean("loadModel", false);

		String modelFilename = bodyJson.getString("modelFilename", null);

		// try {
		if (saveModel) {
			if (modelFilename == null || modelFilename.isEmpty()) {
				rc.response().setStatusCode(500).end("no filename");
				return;
			}
			// service.nn.saveModel(modelFilename + ".zip", true);
			// service.saveObservationData("train", modelFilename, service.trainColoData);
			// service.saveObservationData("test", modelFilename, service.testColorData);
			System.out.println("Saved model: " + modelFilename);
		}

		if (resetModel) {
			// service.createNewDataSets();
			// service.nn.initializeNewModel();
			System.out.println("Reset model");
		}

		if (loadModel) {
			if (modelFilename == null || modelFilename.isEmpty()) {
				rc.response().setStatusCode(500).end("no filename");
				return;
			}
			// service.createNewDataSets();
			// service.nn.restoreModel(modelFilename + ".zip", true);
			// service.loadObservationData("train", modelFilename, service.trainColoData);
			// service.loadObservationData("test", modelFilename, service.testColorData);
			System.out.println("Loaded model: " + modelFilename);
		}

		rc.response().end(new JsonObject().encode());
		// } catch (IOException e) {
		// e.printStackTrace();
		// rc.response().setStatusCode(500).end(e.getMessage());
		// }
	}
}
