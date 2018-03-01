package com.daisyworks.language;

import static org.junit.Assert.assertEquals;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.Test;

public class WordNGramTokenizerTest {

	@Test
	public void testSingleNgrams() {
		Set<String> ngrams = new HashSet<String>();

		List<String> wordList = TokenizeSentenceIntoWords.tokenize("see tom run");
		WordNGramTokenizer.tokenize(wordList, 1, 1, ngrams);
		assertEquals(3, ngrams.size());
	}

	@Test
	public void testDouble() {
		Set<String> ngrams = new HashSet<String>();
		List<String> wordList = TokenizeSentenceIntoWords.tokenize("see tom run");
		WordNGramTokenizer.tokenize(wordList, 2, 2, ngrams);
		assertEquals(2, ngrams.size());
	}

	@Test
	public void testTripple() {
		Set<String> ngrams = new HashSet<String>();
		List<String> wordList = TokenizeSentenceIntoWords.tokenize("see tom run");
		WordNGramTokenizer.tokenize(wordList, 3, 3, ngrams);
		assertEquals(1, ngrams.size());
	}

	@Test
	public void testQuad() {
		Set<String> ngrams = new HashSet<String>();
		List<String> wordList = TokenizeSentenceIntoWords.tokenize("see tom run");
		WordNGramTokenizer.tokenize(wordList, 4, 4, ngrams);
		assertEquals(0, ngrams.size());
	}

	@Test
	public void testOneToTwo() {
		Set<String> ngrams = new HashSet<String>();
		List<String> wordList = TokenizeSentenceIntoWords.tokenize("see tom run");
		WordNGramTokenizer.tokenize(wordList, 1, 2, ngrams);
		assertEquals(5, ngrams.size());
	}

	@Test
	public void testOneToThree() {
		Set<String> ngrams = new HashSet<String>();
		List<String> wordList = TokenizeSentenceIntoWords.tokenize("see tom run");
		WordNGramTokenizer.tokenize(wordList, 1, 3, ngrams);
		assertEquals(6, ngrams.size());
	}

	@Test
	public void testOneToFour() {
		Set<String> ngrams = new HashSet<String>();
		List<String> wordList = TokenizeSentenceIntoWords.tokenize("see tom run");
		WordNGramTokenizer.tokenize(wordList, 1, 4, ngrams);
		assertEquals(6, ngrams.size());
	}
}
