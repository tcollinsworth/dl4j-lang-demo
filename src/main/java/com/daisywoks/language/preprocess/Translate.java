package com.daisywoks.language.preprocess;

import io.vertx.core.json.JsonObject;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.util.Arrays;

//- en english
//- nl dutch
//- es spanish
//- fr french
//- de german
//- it italian
//- cy welsh

public class Translate {

	public static void main(String[] args) throws IOException {
		String s = new String(Files.readAllBytes(new File("src/main/resources/examplesRaw/english/aliceInWonderlandEnglish").toPath()));
		String[] lines = s.split("\n");

		StringBuilder b = new StringBuilder();

		Arrays.asList(lines).stream().filter((line) -> !line.trim().isEmpty()).forEach((line) -> {
			String t = getTranslation(line, "cy");
			String tt = new JsonObject(t).getJsonObject("data").getJsonArray("translations").getJsonObject(0).getString("translatedText");
			// { "data": { "translations": [ { "translatedText": "Hola" } ] }}
				b.append(tt);
				b.append("\n\n");
			});
		System.out.println(b.toString());
		// TODO write to file
	}

	public static String getJson(String lang, String data) {
		String json = String.format("{\"source\": \"en\",\"target\": \"%s\",\"format\": \"text\",\"q\": [\"%s\"]}", lang, data);
		return json;
	}

	public static String getTranslation(String line, String lang) {
		HttpURLConnection con = null;
		DataOutputStream out = null;
		BufferedReader in = null;
		try {
			URL url = new URL("https://translation.googleapis.com/language/translate/v2?key=" + System.getenv("API_KEY"));

			con = (HttpURLConnection) url.openConnection();
			con.setRequestMethod("POST");
			con.setRequestProperty("Content-Type", "application/json");
			con.setDoOutput(true);
			out = new DataOutputStream(con.getOutputStream());
			out.writeBytes(getJson(lang, line));
			out.flush();

			in = new BufferedReader(new InputStreamReader(con.getInputStream()));
			String inputLine;
			StringBuffer content = new StringBuffer();
			while ((inputLine = in.readLine()) != null) {
				content.append(inputLine);
			}
			return content.toString();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				out.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			try {
				in.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			con.disconnect();
		}
		return null;
	}
}
