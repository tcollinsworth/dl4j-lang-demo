package com.daisyworks.language;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

public class WordNGramTokenizer {

	public static void tokenize(List<String> wordListOrdered, int smallest, int longest, Set<String> ngrams) {
		String[] words = wordListOrdered.toArray(new String[0]);

		StringBuilder nGram = new StringBuilder();

		for (int i = 0; i < words.length; i++) {
			for (int ngramSize = smallest; ngramSize <= longest; ngramSize++) {
				// are there enough words remaining for ngramSize
				if (words.length - i < ngramSize) {
					break;
				}
				String[] ngramWordsArray = Arrays.copyOfRange(words, i, i + ngramSize);
				nGram.setLength(0);
				Arrays.asList(ngramWordsArray).forEach((w) -> {
					nGram.append(w);
					nGram.append(' ');
				});
				ngrams.add(nGram.toString().trim());
			}
		}
	}
}
