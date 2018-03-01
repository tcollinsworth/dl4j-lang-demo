package com.daisyworks.language;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;

public class DataSwizzler {
	private final String[] rawExampleDirs;
	private final double valFraction;

	private final Map<String, List<Pair<String, String>>> dataSets;

	public DataSwizzler(String[] rawExampleDirs, double valFraction) {
		this.rawExampleDirs = rawExampleDirs;
		this.valFraction = valFraction;

		// key=train|val|test val=List<Pair<class,ngrams>>
		dataSets = new HashMap<>();
		dataSets.put("train", new ArrayList<>());
		dataSets.put("validation", new ArrayList<>());
	}

	public static void main(String[] args) {
		String[] rawExampleDirs = { //
		"src/main/resources/examplesRaw/dutch", //
				"src/main/resources/examplesRaw/english", //
				"src/main/resources/examplesRaw/french", //
				"src/main/resources/examplesRaw/german", //
				"src/main/resources/examplesRaw/italian", //
				"src/main/resources/examplesRaw/spanish", //
				"src/main/resources/examplesRaw/welsh" //
		};

		DataSwizzler ds = new DataSwizzler(rawExampleDirs, 0.25);
		ds.loadData();
	}

	private void loadData() {
		// TODO read existing train/val/test for consistency if they exist
		// TODO persist dataSets if not read from files for consistency
		// TODO persist longest sequence and classifications

		int[] maxChars = { 0 };

		Arrays.asList(rawExampleDirs).forEach((dir) -> {
			String classification = new File(dir).getName();
			Set<String> ngrams = getNgrams(dir);
			String longestNgram = ngrams.stream().max((a, b) -> Integer.compare(a.length(), b.length())).get();
			maxChars[0] = longestNgram.length() > maxChars[0] ? longestNgram.length() : maxChars[0];
			System.out.printf("%s ngrams %d maxCharLength %d %s\n", classification, ngrams.size(), longestNgram.length(), ngrams);
			randomSplitSampleData(classification, ngrams);

		});
		System.out.printf("maxChars %d\n", maxChars[0]);
		System.out.println(dataSets);
		System.out.printf("train %d, val %d\n", dataSets.get("train").size(), dataSets.get("validation").size());
	}

	// key=train|val|test val=List<Pair<class,ngrams>>
	private void randomSplitSampleData(String classification, Set<String> ngramSet) {
		List<String> ngrams = new ArrayList<>(ngramSet);
		Collections.shuffle(ngrams);

		List<Pair<String, String>> ngramsTrain = dataSets.get("train");
		List<Pair<String, String>> ngramsValidation = dataSets.get("validation");

		int valCnt = (int) (ngrams.size() * valFraction);
		for (int i = 0; i < valCnt; i++) {
			ngramsValidation.add(Pair.of(classification, ngrams.remove(i)));
		}

		ngrams.forEach((ng) -> ngramsTrain.add(Pair.of(classification, ng)));
	}

	private static final Set<String> getNgrams(String dir) {
		FileSentenceIterator fsi = null;
		try {
			fsi = new FileSentenceIterator(new File(dir));
			while (fsi.hasNext()) {
				// iterate sentences
				String sentence = fsi.nextSentence().trim();
				if (sentence == null || sentence.isEmpty()) {
					continue;
				}
				List<String> wordListOrdered = TokenizeSentenceIntoWords.tokenize(sentence);

				Set<String> ngrams = new HashSet<String>();
				WordNGramTokenizer.tokenize(wordListOrdered, 1, 5, ngrams);
				return ngrams;
			}
		} finally {
			if (fsi != null) {
				fsi.finish();
			}
		}
		return Collections.emptySet();
	}
}
