import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

/*
 * 10-605 hw3
 * @author Lingxue Zhu
 * @version 2016/10/06
 */

public class LR {
	// for soft sigmoid
	private static double overflow = 20;

	/**
	 * Train a logistic regression by streaming through training data from
	 * stdin, then predict labels on testing data in the testFile, output is
	 * written to stdout.
	 * 
	 * @param args
	 *                with length 6
	 */
	public static void main(String[] args) throws IOException {
		if (args.length != 6) {
			throw new IllegalArgumentException(
					"Usage: java LR vocSize rate regCoeff maxIter trainSize testFile");
		}
		int vocSize = Integer.valueOf(args[0]);
		double rate = Double.valueOf(args[1]);
		double regCoeff = Double.valueOf(args[2]);
		int maxIter = Integer.valueOf(args[3]);
		int trainSize = Integer.valueOf(args[4]);
		String testFile = args[5];

		// training; one binary classifier for each label
		String targetLabel = "Place";
		Map<Integer, Double> coeffLR = trainLR(targetLabel, vocSize, rate, regCoeff, maxIter, trainSize);

		// prediction on test set
		predictLR(targetLabel, coeffLR, testFile);
	}

	/**
	 * Train a logistic regression using stochastic gradient descent by
	 * streaming through training data from stdin.
	 * 
	 * @param vocSize
	 *                vocabulary size for hashtables using the hash trick
	 * @param rate
	 *                initial learning rate
	 * @param regCoeff
	 *                coefficient for L2 regularization
	 * @param maxIter
	 *                maximum number of iterations for stochastic gradient
	 *                descent
	 * @param trainSize
	 *                the size of training data, i.e., number of training
	 *                samples
	 * @return the hashtable containing coefficients for features
	 */
	public static Map<Integer, Double> trainLR(String targetLabel, int vocSize, double initRate, double regCoeff,
			int maxIter, int trainSize) throws IOException {
		// model coefficients
		Map<Integer, Double> coeffLR = new HashMap<>();
		// keep record for lazy update on regularization
		Map<Integer, Integer> coeffUpdateLag = new HashMap<>();

		// Streaming through trainin data from stdin, update model
		// parameters
		BufferedReader trainDataIn = new BufferedReader(new InputStreamReader(System.in));
		trainDataIn.mark(trainSize); // mark the starting point

		int totalIters = 0;
		for (int iter = 1; iter <= maxIter; iter++) {
			trainDataIn.reset();
			// learning rate decays with iteration
			double rate = initRate / (iter * iter);

			// One pass through training samples
			for (int n = 0; n < trainSize; n++) {
				totalIters += 1;
				String trainDoc = trainDataIn.readLine();
				trainOneDoc(targetLabel, trainDoc, totalIters, coeffLR, coeffUpdateLag, vocSize, rate,
						regCoeff);
			}
		}
		trainDataIn.close();

		return coeffLR;
	}

	private static void trainOneDoc(String targetLabel, String trainDoc, int totalIters,
			Map<Integer, Double> coeffLR, Map<Integer, Integer> coeffUpdateLag, int vocSize, double rate,
			double regCoeff) {
		Vector<String> tokens = tokenizeDoc(trainDoc);
		// response = {0, 1} for the target label
		int response = getResponse(tokens.elementAt(1), targetLabel);

		// count frequency of words (words starting from the 3rd token,
		// index 2)
		Map<Integer, Integer> wordCounts = getWordCounts(tokens, vocSize, 2);

		// update coeffLR and coeffUpdateLag for the appeared words
		updateCoeff(response, wordCounts, coeffLR, coeffUpdateLag, totalIters, rate, regCoeff);
	}

	/**
	 * Update corresponding coefficients for a subset of given wordIDs and
	 * counts.
	 * 
	 * @param response
	 *                binary label
	 * @param wordCounts
	 *                map of words and counts
	 * @param coeffLR
	 *                map of coefficients to be updated
	 * @param coeffUpdateLag
	 *                map of lags for lazy regularizations to be updated
	 * @param totalIters
	 *                keep track of current iteration number
	 * @param rate
	 *                given learning rate
	 * @param regCoeff
	 *                given regularization coefficient
	 */
	private static void updateCoeff(int response, Map<Integer, Integer> wordCounts, Map<Integer, Double> coeffLR,
			Map<Integer, Integer> coeffUpdateLag, int totalIters, double rate, double regCoeff) {
		// prediction using current model parameters
		double predict = getDocPrediction(wordCounts, coeffLR);

		// update corresponding coefficients using word counts
		for (Integer wordID : wordCounts.keySet()) {
			int count = wordCounts.get(wordID);
			// new word
			if (!coeffLR.containsKey(wordID)) {
				coeffLR.put(wordID, 0.0);
				coeffUpdateLag.put(wordID, 0);
			}
			// regularization that would have been performed
			double newcoeff = coeffLR.get(wordID)
					* Math.pow(1 - 2 * rate * regCoeff, totalIters - coeffUpdateLag.get(wordID));
			// gradient descent contributed from data
			newcoeff += rate * (response - predict) * count;
			// update
			coeffLR.put(wordID, newcoeff);
			coeffUpdateLag.put(wordID, totalIters);
		}
	}

	/**
	 * Get prediction of probability for one example using current parameter
	 * for logistic regression.
	 * 
	 * @param wordCounts
	 * @param coeffLR
	 * @return
	 */
	private static double getDocPrediction(Map<Integer, Integer> wordCounts, Map<Integer, Double> coeffLR) {
		double score = 0;
		for (Integer wordID : wordCounts.keySet()) {
			// only use words that have been seen in training
			// samples
			if (coeffLR.containsKey(wordID)) {
				score += wordCounts.get(wordID) * coeffLR.get(wordID);
			}
		}
		return sigmoid(score);
	}

	/**
	 * Predict labels on testing data in the testFile. Output is written to
	 * stdout.
	 * 
	 * @param coeffLR
	 *                trained coefficients for logistic regression
	 * @param testFile
	 *                file name of the test data
	 */
	public static void predictLR(String targetLabel, Map<Integer, Double> coeffLR, String testFile)
			throws IOException {
		// vocabulary size
		int vocSize = coeffLR.size();

		// Streaming through testing data from file
		BufferedReader testDataIn = new BufferedReader(new FileReader(testFile));
		String testDoc;

		while ((testDoc = testDataIn.readLine()) != null) {

			// count frequency of words (words starting from index 2)
			Vector<String> tokens = tokenizeDoc(testDoc);
			Map<Integer, Integer> wordCounts = getWordCounts(tokens, vocSize, 2);
			
			// for debugging
			System.out.print(String.format("docid=%s, labels=%s, ", tokens.elementAt(0), tokens.elementAt(1)));
			
			// prediction and output
			double predict = getDocPrediction(wordCounts, coeffLR);
			System.out.print(String.format("%s\t%f", targetLabel, predict));
			
			System.out.print("\n");
		}

		testDataIn.close();
	}

	/**
	 * Map a word to an ID as its key in hashtables.
	 * 
	 * @param word
	 * @param vocSize
	 *                the size of hashtable
	 * @return an integer between [0, vocSize-1]
	 */
	private static int wordToID(String word, int vocSize) {
		int id = word.hashCode() % vocSize;
		if (id < 0) {
			id += vocSize;
		}
		return id;
	}

	/**
	 * Tokenize document.
	 * 
	 * @param curDoc
	 *                string for the full text in document
	 * @return a list of tokens
	 */
	private static Vector<String> tokenizeDoc(String curDoc) {
		String[] words = curDoc.split("\\s+");
		Vector<String> tokens = new Vector<String>();
		for (int i = 0; i < words.length; i++) {
			words[i] = words[i].replaceAll("\\W", "");
			if (words[i].length() > 0) {
				tokens.add(words[i]);
			}
		}
		return tokens;
	}

	private static Map<Integer, Integer> getWordCounts(Vector<String> tokens, int vocSize, int startIndex) {
		Map<Integer, Integer> wordCounts = new HashMap<>();
		for (int i = startIndex; i < tokens.size(); i++) {
			Integer wordID = wordToID(tokens.elementAt(i), vocSize);
			if (wordCounts.containsKey(wordID)) {
				wordCounts.put(wordID, wordCounts.get(wordID) + 1);
			} else {
				wordCounts.put(wordID, 1);
			}
		}
		return wordCounts;
	}

	/**
	 * Helper function to get binary response from current document
	 * label(s).
	 * 
	 * @param labelString
	 *                a string of potentially multiple labels, separated by
	 *                ","
	 * @param targetLabel
	 *                the target label for the binary classifier
	 * @return response {0,1} indicating whether target label is among the
	 *         list of labels
	 */
	private static int getResponse(String labelString, String targetLabel) {
		String[] labels = labelString.split(",");
		int response = Arrays.asList(labels).contains(targetLabel) ? 1 : 0;
		return response;
	}

	/**
	 * A soft sigmoid function to avoid overflow.
	 * 
	 * @param score
	 *                will be truncated between [-overflow, overflow]
	 * @return probability between [0,1] using sigmoid(score)
	 */
	private static double sigmoid(double score) {
		if (score > overflow)
			score = overflow;
		else if (score < -overflow)
			score = -overflow;
		double exp = Math.exp(score);
		return exp / (1 + exp);
	}

}