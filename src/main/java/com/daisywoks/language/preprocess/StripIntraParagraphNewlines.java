package com.daisywoks.language.preprocess;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;

public class StripIntraParagraphNewlines {
	public static void main(String[] args) throws IOException {
		String s = new String(Files.readAllBytes(new File("src/main/resources/examplesRaw/english/aliceAndWonderland").toPath()));
		s = s.replace("--", " ");
		String[] lines = s.split("\n|\r");
		StringBuilder sb = new StringBuilder();
		Arrays.asList(lines).forEach((line -> {
			line = line.trim();
			if (line.isEmpty()) {
				sb.append("\n\n");
			} else {
				sb.append(line);
				sb.append(" ");
			}
		}));
		System.out.println(sb.toString());
	}
}
