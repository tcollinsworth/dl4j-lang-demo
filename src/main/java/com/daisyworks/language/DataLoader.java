package com.daisyworks.language;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.daisyworks.demo.Service;

public class DataLoader {
	private static final int IDX = 0;
	private static final int VAL = 1;

	private final Service svc;

	public DataLoader(Service svc) {
		this.svc = svc;
	}

	public void loadDataSetStats() throws IOException {
		String s = new String(Files.readAllBytes(new File("src/main/resources/examples/dataSetStats.txt").toPath()));
		Arrays.asList(s.split("\n")).forEach((l) -> {
			String[] parts = l.split(":");
			if (parts[0].equals("maxExampleLength")) {
				svc.maxExampleLength = Integer.parseInt(parts[1]);
			}
		});
	}

	public void loadOutputClassificationSet() throws IOException {
		List<String> classes = new ArrayList<>();

		String s = new String(Files.readAllBytes(new File("src/main/resources/examples/classificationMap.txt").toPath()));
		Arrays.asList(s.split("\n")).forEach((l) -> {
			String[] parts = l.split(":");

			classes.add(parts[VAL]);
			svc.classificationNameMap.put(parts[VAL], Integer.parseInt(parts[IDX]));
		});
		svc.classificationSet = classes.toArray(new String[0]);
		svc.outputClassificationCnt = classes.size();
	}

	public void loadInputCharacterSet() throws IOException {
		List<Character> chars = new ArrayList<>();

		String s = new String(Files.readAllBytes(new File("src/main/resources/examples/charMap.txt").toPath()));
		Arrays.asList(s.split("\n")).forEach((l) -> {
			String[] parts = l.split(":");
			chars.add((char) Integer.parseInt(parts[VAL]));
			svc.charValMap.put((char) Integer.parseInt(parts[VAL]), Integer.parseInt(parts[IDX]));
		});
		svc.characterSet = chars.toArray(new Character[0]);
		svc.inputFeatureCnt = chars.size();
	}

	public void loadDataSets() {
		svc.trainDataSetIterator = new ParagraphFileExampleIterator("src/main/resources/examples/train", svc.maxExampleLength, svc.charValMap, svc.classificationSet,
				svc.miniBatchSize);

		svc.validationDataSetIterator = new ParagraphFileExampleIterator("src/main/resources/examples/validation", svc.maxExampleLength, svc.charValMap, svc.classificationSet,
				svc.miniBatchSize);

		svc.testDataSetIterator = new ParagraphFileExampleIterator("src/main/resources/examples/test", svc.maxExampleLength, svc.charValMap, svc.classificationSet,
				svc.miniBatchSize);

		svc.trainDataSetIterator.next(100);
		svc.validationDataSetIterator.next(100);
		svc.testDataSetIterator.next(100);
	}
}
