package com.daisyworks.demo;

import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.RoutingContext;

import com.daisyworks.demo.model.Inferrer.Output;
import com.daisyworks.demo.model.NeuralNet.Observation;

/**
 * @author troy
 *
 */
public class ColorRequestHandler extends RequestHandler {

	public ColorRequestHandler(RoutingContext rc, Service service) {
		super(rc, service);
	}

	// discriminator is constant and not currently being used and is not necessary
	public static final float ColorDiscriminator = 0.1f;
	public static final float RgbScale = 1f / 255f;

	public static enum Colors {
		white, grey, black, brown, red, orange, yellow, green, blue, violet, pink
	};

	@Override
	public void handle() {
		// create observation
		JsonArray rgbJsonArray = bodyJson.getJsonArray("rgb");
		float[] scaledRGB = getScaledInputs(rgbJsonArray);
		int classificationIdx = Colors.valueOf(bodyJson.getString("color")).ordinal();
		String color = Colors.values()[classificationIdx].name();
		Observation observation = service.nn.new Observation(classificationIdx, ColorDiscriminator, scaledRGB[0], scaledRGB[1], scaledRGB[2], 0);
		// System.out.println(String.format("observation, %s, %s", rgbJsonArray, color));

		boolean train = bodyJson.getBoolean("train");
		if (train) {
			// randomly assign to training or test/validation windowed set
			if (Math.round(Math.random()) == 0) {
				service.trainColoData.addObservation(observation);
			} else {
				service.testColorData.addObservation(observation);
			}

			service.trainer.train(service);
		}

		String stats = service.evaluator.grade(service);

		JsonObject respObj = getColorInference(service, bodyJson);
		respObj.put("stats", stats);

		rc.response().end(respObj.encode());
	}

	private float[] getScaledInputs(JsonArray rgbJsonArray) {
		int red = rgbJsonArray.getInteger(0);
		int green = rgbJsonArray.getInteger(1);
		int blue = rgbJsonArray.getInteger(2);

		// System.out.println("rgb: " + red + "," + green + "," + blue);

		float[] scaledRGB = new float[3];

		scaledRGB[0] = red * RgbScale;
		scaledRGB[1] = green * RgbScale;
		scaledRGB[2] = blue * RgbScale;

		// System.out.println("scaled rgb: " + Arrays.toString(scaledRGB));

		return scaledRGB;
	}

	private JsonObject getColorInference(Service service, JsonObject bodyJson) {
		JsonObject nextColorJsonObj = bodyJson.getJsonObject("nextColor");
		JsonArray rgbJsonArray = nextColorJsonObj.getJsonArray("rgb");
		float[] scaledRGB = getScaledInputs(rgbJsonArray);

		Observation inputs = service.nn.new Observation(ColorDiscriminator, scaledRGB[0], scaledRGB[1], scaledRGB[2], 0);

		Output output = service.inferrer.infer(inputs);

		int classificationIdx = output.outputs[0];
		String color = Colors.values()[classificationIdx].name();

		JsonObject respObj = new JsonObject();
		respObj.put("color", color);
		JsonArray classProbabilities = new JsonArray(output.classificationProbabilities);
		respObj.put("colorProbabilities", classProbabilities);
		respObj.put("timeMs", output.timeMs);

		// System.out.println(String.format("reponse: %s, %s, %s", respObj.encode(), rgbJsonArray, color));

		return respObj;
	}
}