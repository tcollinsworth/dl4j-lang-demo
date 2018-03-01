package com.daisyworks.language;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.file.Files;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DataSwizzler {
	private static final Logger Log = LoggerFactory.getLogger(DataSwizzler.class);

	private static final int DataSetNameIdx = 0;
	private static final int ClassificationIdx = 1;
	private static final int NgramIdx = 2;

	private static final String newExamplesDirRelativePath = "src/main/resources/examples";
	private static final String dataSetStatsFileName = newExamplesDirRelativePath + File.separator + "dataSetStats.txt";
	private static final String dataSetsFileName = newExamplesDirRelativePath + File.separator + "dataSets.txt";
	private static final String classificationsFileName = newExamplesDirRelativePath + File.separator + "classifications.txt";

	private final String examplesRawRootDir;
	private final double validationRatio;

	private final int minNgramWords;
	private final int maxNgramWords;

	private final Set<String> classifications = new HashSet<>();
	private final Map<String, List<Pair<String, String>>> dataSets;

	private int examplesMaxCharLength;

	public DataSwizzler(String examplesRawRootDir, double validationRatio, int minNgramWords, int maxNgramWords) {
		this.examplesRawRootDir = examplesRawRootDir;
		this.validationRatio = validationRatio;
		this.minNgramWords = minNgramWords;
		this.maxNgramWords = maxNgramWords;

		// key=train|val|test val=List<Pair<class,ngrams>>
		dataSets = new HashMap<>();
		dataSets.put("train", new ArrayList<>());
		dataSets.put("validation", new ArrayList<>());
	}

	public static void main(String[] args) throws FileNotFoundException, IOException {
		DataSwizzler ds = new DataSwizzler("src/main/resources/examplesRaw", 0.25, 1, 5);
		ds.loadData();
	}

	private void loadData() throws FileNotFoundException, IOException {
		File rootDir = new File(newExamplesDirRelativePath);
		if (!rootDir.exists()) {
			rootDir.mkdirs();
		}

		if (!loadPriorExamples()) {
			discoverPersistRawExampleDirsClassifications();
			loadParsePersistRawExamples();
		}

		Log.info("Classifications {}", classifications);
		Log.info("Examples max char length {}", examplesMaxCharLength);
		dataSets.forEach((dataSetName, dataSet) -> {
			Log.info("DataSet {} Examples {}", dataSetName, dataSet.size());
		});
	}

	private void discoverPersistRawExampleDirsClassifications() throws FileNotFoundException, IOException {
		File exampleDir = new File(examplesRawRootDir);
		File[] exampleEntries = exampleDir.listFiles();
		Arrays.stream(exampleEntries).filter((entry) -> entry.isDirectory()).forEach((dir) -> {
			classifications.add(dir.getName());
		});

		persistClassifications();
	}

	private void persistClassifications() throws FileNotFoundException, IOException {
		File examplesDataSetStatsFile = new File(classificationsFileName);

		try (BufferedWriter w = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(examplesDataSetStatsFile)))) {
			classifications.forEach((c) -> {
				try {
					w.write(String.format("%s\n", c));
				} catch (Exception e) {
					e.printStackTrace();
				}
			});
		}
	}

	private boolean loadPriorExamples() throws IOException {
		File dataSetsFile = new File(dataSetsFileName);
		File dataSetStatsFile = new File(dataSetStatsFileName);
		File classificationsFile = new File(classificationsFileName);

		if (!dataSetsFile.exists() || !dataSetStatsFile.exists() || !classificationsFile.exists()) {
			return false;
		}

		Log.info("Loading existing parsed, split examples");

		loadClassifications();
		loadDataSetStats();
		loadDataSets();

		return true;
	}

	private void loadClassifications() throws IOException {
		String s = new String(Files.readAllBytes(new File(classificationsFileName).toPath()));
		String[] classes = s.split("\n");
		Arrays.asList(classes).forEach((c) -> {
			classifications.add(c);
		});
	}

	private void loadDataSets() throws IOException {
		String s = new String(Files.readAllBytes(new File(dataSetsFileName).toPath()));
		String[] examples = s.split("\n");

		String curDataSetName = "validation";
		List<Pair<String, String>> curDataSet = dataSets.get(curDataSetName);

		for (int i = 1; i < examples.length; i++) {
			String[] parts = examples[i].split(":");
			if (!curDataSetName.equals(parts[DataSetNameIdx])) {
				curDataSetName = parts[DataSetNameIdx];
				curDataSet = dataSets.get(curDataSetName);
			}
			curDataSet.add(Pair.of(parts[ClassificationIdx], parts[NgramIdx]));
		}
	}

	public void loadDataSetStats() throws IOException {
		String s = new String(Files.readAllBytes(new File(dataSetStatsFileName).toPath()));
		Arrays.asList(s.split("\n")).forEach((l) -> {
			String[] parts = l.split(":");
			if (parts[0].equals("maxExampleLength")) {
				examplesMaxCharLength = Integer.parseInt(parts[1]);
			}
		});
	}

	private void loadParsePersistRawExamples() throws FileNotFoundException, IOException {
		Log.info("Loading, parsing, splitting, persisting new examples");
		Arrays.asList(examplesRawRootDir).forEach((dir) -> {
			String classification = new File(dir).getName();
			Set<String> ngrams = getNgrams(dir, minNgramWords, maxNgramWords);
			String longestNgram = ngrams.stream().max((a, b) -> Integer.compare(a.length(), b.length())).get();
			examplesMaxCharLength = longestNgram.length() > examplesMaxCharLength ? longestNgram.length() : examplesMaxCharLength;
			randomSplitSampleData(classification, ngrams);
		});
		persistDataSetStats(examplesMaxCharLength);
		persistDataSets();
	}

	private void persistDataSets() throws FileNotFoundException, IOException {
		File examplesDataSetStatsFile = new File(newExamplesDirRelativePath + File.separator + "dataSets.txt");

		try (BufferedWriter w = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(examplesDataSetStatsFile)))) {
			w.write(String.format("dataset,classification,ngram\n"));

			dataSets.forEach((dataset, ngrams) -> {
				ngrams.forEach((pair) -> {
					try {
						w.write(String.format("%s:%s:%s\n", dataset, pair.getKey(), pair.getValue()));
					} catch (Exception e) {
						e.printStackTrace();
					}
				});
			});
		}
	}

	// maxExampleLength for truncating/padding
	private static void persistDataSetStats(int maxExampleLength) throws FileNotFoundException, IOException {
		File examplesDataSetStatsFile = new File(newExamplesDirRelativePath + File.separator + "dataSetStats.txt");

		try (BufferedWriter w = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(examplesDataSetStatsFile)))) {
			w.write(String.format("%s:%d\n", "maxExampleLength", maxExampleLength));
		}
	}

	// key=train|val|test val=List<Pair<class,ngrams>>
	private void randomSplitSampleData(String classification, Set<String> ngramSet) {
		List<String> ngrams = new ArrayList<>(ngramSet);
		Collections.shuffle(ngrams);

		List<Pair<String, String>> ngramsTrain = dataSets.get("train");
		List<Pair<String, String>> ngramsValidation = dataSets.get("validation");

		int valCnt = (int) (ngrams.size() * validationRatio);
		for (int i = 0; i < valCnt; i++) {
			ngramsValidation.add(Pair.of(classification, ngrams.remove(i)));
		}

		ngrams.forEach((ng) -> ngramsTrain.add(Pair.of(classification, ng)));
	}

	private static final Set<String> getNgrams(String dir, int minNgramWords, int maxNgramWords) {
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
				WordNGramTokenizer.tokenize(wordListOrdered, minNgramWords, maxNgramWords, ngrams);
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
