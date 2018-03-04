package com.daisyworks.demo;

import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.RoutingContext;

/**
 * @author troy
 *
 */
public class LangRequestHandler extends RequestHandler {

	public LangRequestHandler(RoutingContext rc, Service service) {
		super(rc, service);
	}

	@Override
	public void handle() {
		// String stats = service.evaluator.grade(service);

		JsonObject respObj = getLangInference(service, bodyJson);
		// respObj.put("stats", stats);

		rc.response().end(respObj.encode());
	}

	private JsonObject getLangInference(Service service, JsonObject bodyJson) {
		String rawExample = bodyJson.getString("example");

		// Observation inputs = service.nn.new Observation(ColorDiscriminator, scaledRGB[0], scaledRGB[1], scaledRGB[2],
		// 0);

		// Output output = service.inferrer.infer(inputs);

		// int classificationIdx = output.outputs[0];
		// String color = Colors.values()[classificationIdx].name();

		JsonObject respObj = new JsonObject();
		// respObj.put("lang", lang);
		respObj.put("lang", "English");
		// JsonArray classProbabilities = new JsonArray(output.classificationProbabilities);
		// respObj.put("colorProbabilities", classProbabilities);
		// respObj.put("timeMs", output.timeMs);

		// System.out.println(String.format("reponse: %s, %s, %s", respObj.encode(), rgbJsonArray, color));

		return respObj;
	}
}