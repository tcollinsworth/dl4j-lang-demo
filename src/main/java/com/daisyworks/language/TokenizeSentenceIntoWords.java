package com.daisyworks.language;

import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

public class TokenizeSentenceIntoWords {
	public static List<String> tokenize(String sentence) {
		List<String> wordList = new ArrayList<>();

		// strip numbers
		sentence = sentence.replaceAll("[0-9]", "");

		// split words - stripping punctuation
		StringTokenizer t = new StringTokenizer(sentence, " \t\n\r\f,.:;?![]\"()¿¡{}«»“”‘’—-*~@#$%^&=+<>/");

		while (t.hasMoreTokens()) {
			String word = t.nextToken().trim();

			// strip leading and trailing quotes
			word = word.replaceAll("^'|'$", "");

			if (word == null || word.isEmpty()) {
				continue;
			}

			wordList.add(word);
		}
		return wordList;
	}

}
