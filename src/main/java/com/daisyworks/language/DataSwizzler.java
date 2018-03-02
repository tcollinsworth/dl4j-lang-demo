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

	private static final int IDX = 0;
	private static final int VAL = 1;

	private static final int DataSetNameIdx = 0;
	private static final int ClassificationIdx = 1;
	private static final int NgramIdx = 2;

	private static final String newExamplesDirRelativePath = "src/main/resources/examples";
	private static final String dataSetStatsFileName = newExamplesDirRelativePath + File.separator + "dataSetStats.txt";
	private static final String dataSetsFileName = newExamplesDirRelativePath + File.separator + "dataSets.txt";
	private static final String classificationsFileName = newExamplesDirRelativePath + File.separator + "classifications.txt";
	private static final String charMapFileName = newExamplesDirRelativePath + File.separator + "charMap.txt";

	private final String examplesRawRootDir;
	private final double validationRatio;

	private final int minNgramWords;
	private final int maxNgramWords;

	private final Set<String> classifications = new HashSet<>();
	private final Map<String, List<Pair<String, String>>> dataSets;

	private final Set<Integer> allCharInts = new HashSet<Integer>();
	// key = char (int), val = char sorted order index from 0-n for all characters in training set
	// encoded to uniform space eliminating gaps and outliers
	private final Map<Character, Double> charMap = new HashMap<Character, Double>();

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

	public void loadData() throws FileNotFoundException, IOException {
		File rootDir = new File(newExamplesDirRelativePath);
		if (!rootDir.exists()) {
			rootDir.mkdirs();
		}

		if (!loadPriorExamples()) {
			discoverPersistRawExampleDirsClassifications();
			loadParsePersistRawExamples();
			persistCharMap();
			loadCharMap();
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

	private void loadCharMap() throws IOException {
		List<Character> chars = new ArrayList<>();

		String s = new String(Files.readAllBytes(new File(charMapFileName).toPath()));
		Arrays.asList(s.split("\n")).forEach((l) -> {
			String[] parts = l.split(":");
			chars.add((char) Integer.parseInt(parts[VAL]));
			charMap.put((char) Double.parseDouble(parts[VAL]), Double.parseDouble(parts[IDX]));
		});
		scaleCharMap();
	}

	// scale from 0-1 for learning/inference efficiency
	private void scaleCharMap() {
		final double scaleFactor = 1d / (charMap.size() - 1d);
		charMap.entrySet().forEach((e) -> {
			e.setValue(e.getValue() * scaleFactor);
		});

	}

	private void persistCharMap() throws FileNotFoundException, IOException {
		Log.info("unique char count={}", allCharInts.size());
		Integer[] uniquChars = allCharInts.toArray(new Integer[0]);
		Arrays.sort(uniquChars);
		Log.info("unique chars={}", Arrays.toString(uniquChars));
		// Create and persist map of chars for input vector - needs to be reusable.
		// 10 - 255, then all higher map as discovered

		File examplesCharMapFile = new File(charMapFileName);

		try (BufferedWriter w = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(examplesCharMapFile)))) {
			// Only persist chars we've trained on, best to exclude untrained chars on inference
			// for (int i = 10; i <= 255; i++) {
			// w.write(String.format("%d:%d\n", i - 10, i));
			// }
			int[] i = { 0 };
			// Arrays.asList(uniquChars).stream().filter((c) -> c > 255).forEach((c) -> persistChar(++i[0], c, w));
			Arrays.asList(uniquChars).forEach((c) -> persistChar(++i[0], c, w));
		}
	}

	private void persistChar(int i, Integer c, BufferedWriter w) {
		try {
			// w.write(String.format("%d:%d\n", i - 10, c));
			w.write(String.format("%d:%d\n", i, c));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private boolean loadPriorExamples() throws IOException {
		File dataSetsFile = new File(dataSetsFileName);
		File dataSetStatsFile = new File(dataSetStatsFileName);
		File classificationsFile = new File(classificationsFileName);
		File charMapFile = new File(charMapFileName);

		if (!dataSetsFile.exists() || !dataSetStatsFile.exists() || !classificationsFile.exists() || !charMapFile.exists()) {
			return false;
		}

		Log.info("Loading existing parsed, split examples");

		loadClassifications();
		loadDataSetStats();
		loadDataSets();
		loadCharMap();

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

	private void accumulateUniqueChars(String sequence) {
		sequence.chars().forEach((c) -> allCharInts.add(c));
	}

	private void loadDataSetStats() throws IOException {
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
		File examplesRawRootDirFile = new File(examplesRawRootDir);
		Log.info("{}", Arrays.asList(examplesRawRootDirFile.list()));
		Arrays.asList(examplesRawRootDirFile.list()).stream() //
				.filter((classificationDir) -> new File(examplesRawRootDir + File.separator + classificationDir).isDirectory()) //
				.forEach((classificationDir) -> {
					Set<String> ngrams = getNgrams(examplesRawRootDir + File.separator + classificationDir, minNgramWords, maxNgramWords);
					String longestNgram = ngrams.stream().max((a, b) -> Integer.compare(a.length(), b.length())).get();
					examplesMaxCharLength = longestNgram.length() > examplesMaxCharLength ? longestNgram.length() : examplesMaxCharLength;
					randomSplitSampleData(classificationDir, ngrams);
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

	private final Set<String> getNgrams(String dir, int minNgramWords, int maxNgramWords) {
		FileSentenceIterator fsi = null;
		try {
			Set<String> ngrams = new HashSet<String>();
			fsi = new FileSentenceIterator(new File(dir));
			while (fsi.hasNext()) {
				// iterate sentences
				String sentence = fsi.nextSentence().trim().toLowerCase();
				if (sentence == null || sentence.isEmpty()) {
					continue;
				}
				accumulateUniqueChars(sentence);
				List<String> wordListOrdered = TokenizeSentenceIntoWords.tokenize(sentence);
				SentenceNgramTokenizer.tokenize(wordListOrdered, minNgramWords, maxNgramWords, ngrams);
				// WordNGramTokenizer.tokenize(wordListOrdered, minNgramWords, maxNgramWords, ngrams);
			}
			return ngrams;
		} finally {
			if (fsi != null) {
				fsi.finish();
			}
		}
	}

	public int getMaxCharLength() {
		return examplesMaxCharLength;
	}

	public List<Pair<String, String>> getDataSet(String dataSetName) {
		return dataSets.get(dataSetName);
	}

	public Set<String> getClassificationSet() {
		return classifications;
	}

	public Map<Character, Double> getCharMap() {
		return charMap;
	}
}
