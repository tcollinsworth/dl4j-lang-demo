package com.daisyworks.preprocess;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Iterates src/main/resources/examplesRaw dirs. Creates a src/main/resources/examples/train dir In each dir, parses
 * multiDoc on empty line creating a new file with index and .txt, i.e., english0.txt, english1.txt Copies non-blank
 * lines (paragraph) till next blank line into new files.
 * 
 * Stats counts size of each file in chars and reports min/max/average. so we know masking required before
 * classification ClassStats tracks all characters in file and reports so we know size of input vector
 * 
 * @author troy
 *
 */
public class SplitMultiDoc {
	static String newExamplesDirRelativePath = "src/main/resources/examples";
	static double trainPct = 0.5;
	static double testPct = 0.25;
	static double valPct = 0.25;

	static SecureRandom rnd = new SecureRandom();

	static List<ClassStats> classesStats = new ArrayList<ClassStats>();

	public static void main(String[] args) throws FileNotFoundException, IOException {
		// remove old src/main/resources/examples
		File examplesDir = new File(newExamplesDirRelativePath);
		if (examplesDir.exists()) {
			examplesDir.delete();
		}

		// create output dirs src/main/resources/examples/train, test, validation
		// model training data
		File newExamplesTrainDir = new File(newExamplesDirRelativePath + File.separator + "train");
		newExamplesTrainDir.mkdirs();

		// for tuning modely hyper-parameters and determine training stopping point
		File newExamplesValDir = new File(newExamplesDirRelativePath + File.separator + "validation");
		newExamplesValDir.mkdirs();

		// asses performance/error rate - MUST NOT USE for tuning model further
		File newExamplesTestDir = new File(newExamplesDirRelativePath + File.separator + "test");
		newExamplesTestDir.mkdirs();

		File exampleDir = new File("src/main/resources/examplesRaw");
		File[] exampleEntries = exampleDir.listFiles();
		Arrays.stream(exampleEntries).filter((entry) -> entry.isDirectory()).forEach((dir) -> {
			System.out.println(dir);
			// Iterate dirs
				processDir(dir, newExamplesTrainDir);
			});

		Set<Integer> allCharInts = new HashSet<Integer>();
		int[] maxCharCnt = { 0 };
		int[] maxCharInt = { 0 };
		classesStats.forEach((cs) -> {
			System.out.println(cs);
			if (cs.charCnt > maxCharCnt[0]) {
				maxCharCnt[0] = cs.charCnt;
			}
			int maxClassCharInt = cs.getMaxCharInt();
			if (maxClassCharInt > maxCharInt[0]) {
				maxCharInt[0] = maxClassCharInt;
			}
			cs.charInts.forEach((i) -> {
				allCharInts.add(i);
			});
		});
		System.out.println("maxCharCnt=" + maxCharCnt[0]);
		System.out.println("maxCharInt=" + maxCharInt[0]);
		// System.out.println("unique chars=" + allCharInts);
		System.out.println("unique char count=" + allCharInts.size());
		Integer[] uniquChars = allCharInts.toArray(new Integer[0]);
		Arrays.sort(uniquChars);
		System.out.println("unique chars=" + Arrays.toString(uniquChars));
		// Create and persist map of chars for input vector - needs to be reusable.
		// 32 - 255, then all higher map as discovered
		persistUniqueCharMap(uniquChars);
	}

	private static void persistUniqueCharMap(Integer[] uniquChars) throws FileNotFoundException, IOException {
		File examplesCharMapFile = new File(newExamplesDirRelativePath + File.separator + "charMap.txt");

		try (BufferedWriter w = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(examplesCharMapFile)))) {
			for (int i = 32; i <= 255; i++) {
				w.write(String.format("%d:%d\n", i, i));
			}
			int[] i = { 255 };
			Arrays.asList(uniquChars).stream().filter((c) -> c > 255).forEach((c) -> persistChar(++i[0], c, w));
		}
	}

	private static void persistChar(Integer i, Integer c, BufferedWriter w) {
		try {
			w.write(String.format("%d:%d\n", i, c));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static void processDir(File dir, File newExamplesDir) {
		// Open files
		File multiDoc = new File(dir.getAbsoluteFile() + File.separator + "multiDoc");
		System.out.println(multiDoc);
		ClassStats classStats = new ClassStats(dir.getName());
		classesStats.add(classStats);

		BufferedWriter w = null;
		try (BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(multiDoc)))) {
			int exampleNo = 0;
			File newExampleFile = createExampleFile(dir, exampleNo);
			ExampleStats exampleStats = new ExampleStats(newExampleFile.getName());
			classStats.examplesStats.add(exampleStats);
			w = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(newExampleFile)));
			String line = null;
			do {
				line = r.readLine();
				if (line == null) {
					break;
				}
				// split on blank line
				// write to separate files all in train named with class
				if (line.isEmpty()) {
					w.close();
					newExampleFile = createExampleFile(dir, ++exampleNo);
					exampleStats = new ExampleStats(newExampleFile.getName());
					classStats.examplesStats.add(exampleStats);
					w = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(newExampleFile)));
					continue;
				}
				w.write(line);
				classStats.addChars(line, exampleStats);
				w.write(System.lineSeparator());
			} while (line != null);
			sampleValTestSets(dir.getName(), exampleNo + 1);
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (w != null) {
				try {
					w.close();
				} catch (Exception e) {
					e.printStackTrace();
					// Ignore
				}
			}
		}
	}

	private static void sampleValTestSets(String classifier, int examples) {
		String sourceDir = newExamplesDirRelativePath + File.separator + "train" + File.separator;
		// System.out.println("sourceDir " + sourceDir);

		Set<Integer> prior = new HashSet<Integer>();

		String valDir = newExamplesDirRelativePath + File.separator + "validation" + File.separator;
		// System.out.println("valDir " + valDir);
		moveSamples(classifier, sourceDir, valDir, examples, valPct, prior);

		String testDir = newExamplesDirRelativePath + File.separator + "test" + File.separator;
		// System.out.println("testDir " + testDir);
		moveSamples(classifier, sourceDir, testDir, examples, valPct, prior);
	}

	private static void moveSamples(String classifier, String sourceDir, String destDir, int examples, double samplePct, Set<Integer> prior) {
		int targetSamples = (int) (((double) examples) * samplePct);
		int curSamples = 0;
		while (true) {
			int sampleNo;
			while (true) {
				sampleNo = rnd.nextInt(examples);
				if (prior.add(sampleNo)) {
					// break when new sampleNo
					break;
				}
			}
			File srcFile = new File(sourceDir + File.separator + classifier + sampleNo + ".txt");
			File dstFile = new File(destDir + File.separator + classifier + sampleNo + ".txt");
			System.out.println(String.format("%s, %s", srcFile, dstFile));
			srcFile.renameTo(dstFile);
			++curSamples;
			if (curSamples >= targetSamples) {
				break;
			}
		}
		System.out.println(String.format("%s, %s, %s, %d, %f, %d, %d", classifier, sourceDir, destDir, examples, samplePct, targetSamples, curSamples));
	}

	// make separate dirs for each class so like the expected data
	private static File createExampleFile(File dir, int exampleNo) {
		File newFile = new File(newExamplesDirRelativePath + File.separator + "train" + File.separator + dir.getName() + exampleNo + ".txt");
		System.out.println(newFile);
		return newFile;
	}

	static class ClassStats {
		public final String classLabel;
		public int charCnt;
		public int examplesMinCharCnt = Integer.MAX_VALUE;
		public int examplesMaxCharCnt;

		public Set<Integer> charInts = new HashSet<Integer>();
		public List<ExampleStats> examplesStats = new ArrayList<ExampleStats>();

		public ClassStats(String classLabel) {
			this.classLabel = classLabel;
		}

		public int getMaxCharInt() {
			int[] maxCharInt = { 0 };
			charInts.forEach((i) -> {
				if (i > maxCharInt[0]) {
					maxCharInt[0] = i;
				}
			});
			return maxCharInt[0];
		}

		public void addChars(String line, ExampleStats exampleStats) {
			line.chars().forEach((i) -> {
				charInts.add(i);
				++charCnt;
				++exampleStats.charCnt;
			});
		}

		public void computeMinMaxExampleCharCnt() {
			if (examplesMaxCharCnt != 0) {
				return;
			}

			examplesStats.forEach((s) -> {
				if (s.charCnt < examplesMinCharCnt) {
					examplesMinCharCnt = s.charCnt;
				}
				if (s.charCnt > examplesMaxCharCnt) {
					examplesMaxCharCnt = s.charCnt;
				}
			});
		}

		@Override
		public String toString() {
			computeMinMaxExampleCharCnt();
			return "ClassStats [classLabel=" + classLabel + ", exampleCnt=" + examplesStats.size() + ", examplesMinCharCnt=" + examplesMinCharCnt + ", examplesMaxCharCnt="
					+ examplesMaxCharCnt + ", charCnt=" + charCnt + ", chars=" + charInts + ", examplesStats=" + examplesStats.toString() + "]";
		}
	}

	static class ExampleStats {
		public final String example;
		public int charCnt;

		public ExampleStats(String example) {
			this.example = example;
		}

		@Override
		public String toString() {
			return "ExampleStats [example=" + example + ", charCnt=" + charCnt + "]";
		}
	}
}
