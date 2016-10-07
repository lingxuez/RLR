import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
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

		// 17 binary classifiers, one for each label
		String[] targetLabels = { "CelestialBody", "Biomolecule", "Organisation", "Work", "Agent", "Event",
				"Person", "ChemicalSubstance", "Place", "Location", "SportsSeason", "Activity",
				"Device", "TimePeriod", "other", "MeanOfTransportation", "Species" };

		// training; one binary classifier for each label
		List<Map<Integer, Double>> coeffLRs = trainLR(targetLabels, vocSize, rate, regCoeff, maxIter,
				trainSize);

		// prediction on test set; get probabilities for all labels
		predictLR(targetLabels, coeffLRs, testFile, vocSize);
	}

	/**
	 * Train the logistic regression using stochastic gradient descent by
	 * streaming through training data from stdin. Train one binary
	 * classifier for each label separately.
	 * 
	 * @param targetLabels
	 *                names of all classifiers to be trained
	 * @param vocSize
	 *                vocabulary size for hashtables using the hash trick
	 * @param initRate
	 *                initial learning rate
	 * @param regCoeff
	 *                coefficient for L2 regularization
	 * @param maxIter
	 *                maximum number of iterations for stochastic gradient
	 *                descent
	 * @param trainSize
	 *                the size of training data, i.e., number of training
	 *                samples
	 * @return the hashtables containing coefficients for features, one set
	 *         for each label
	 */
	public static List<Map<Integer, Double>> trainLR(String[] targetLabels, int vocSize, double initRate,
			double regCoeff, int maxIter, int trainSize) throws IOException {
		int labelNumber = targetLabels.length;
		// <label, model coefficients>
		List<Map<Integer, Double>> coeffLRs = new ArrayList<Map<Integer, Double>>(labelNumber);
		// <label, lag_iters>, for lazy update on regularization
		List<Map<Integer, Integer>> coeffUpdateLags = new ArrayList<Map<Integer, Integer>>(labelNumber);
		// Initialization
		for (int i = 0; i < labelNumber; i++) {
			coeffLRs.add(new HashMap<Integer, Double>());
			coeffUpdateLags.add(new HashMap<Integer, Integer>());
		}

		// Streaming through training data from stdin
		BufferedReader trainDataIn = new BufferedReader(new InputStreamReader(System.in));
		int totalIters = 0;

		for (int iter = 1; iter <= maxIter; iter++) {
			double rate = initRate / (iter * iter); // learning rate

			// One pass through training samples
			for (int n = 0; n < trainSize; n++) {
				totalIters += 1;
				String trainDoc = trainDataIn.readLine();
				Vector<String> tokens = tokenizeDoc(trainDoc);
				String trainLabel = tokens.elementAt(1);
				Map<Integer, Integer> wordCounts = getWordCounts(tokens, vocSize, 2);

				// update all binary classifiers
				for (int i = 0; i < labelNumber; i++) {
					trainOneDoc(targetLabels[i], trainLabel, wordCounts, totalIters,
							coeffLRs.get(i), coeffUpdateLags.get(i), vocSize, rate,
							regCoeff);
				}
			}

			// lazy update for regularization
			for (int i = 0; i < labelNumber; i++) {
				updateRegularization(coeffLRs.get(i), coeffUpdateLags.get(i), regCoeff, rate,
						totalIters);
			}
		}
		trainDataIn.close();

		return coeffLRs;
	}

	/**
	 * In the end of each epoch, perform all lagged updates for
	 * regularizations.
	 * 
	 * @param coeffLR
	 *                model coefficients
	 * @param coeffUpdateLag
	 *                lagged iters
	 * @param regCoeff
	 *                regularization coefficients
	 * @param rate
	 *                learning rate
	 * @param totalIters
	 *                current iter number
	 */
	private static void updateRegularization(Map<Integer, Double> coeffLR, Map<Integer, Integer> coeffUpdateLag,
			double regCoeff, double rate, int totalIters) {
		for (Integer wordID : coeffLR.keySet()) {
			// regularization that would have been performed
			double newcoeff = coeffLR.get(wordID)
					* Math.pow(1 - 2 * rate * regCoeff, totalIters - coeffUpdateLag.get(wordID));
			// update
			coeffLR.put(wordID, newcoeff);
			coeffUpdateLag.put(wordID, totalIters);
		}
	}

	/**
	 * Update the coefficients for a given classifier using gradient descent
	 * from one document.
	 * 
	 * @param targetLabel
	 *                the label for the classifier to be updated
	 * @param trainLabel
	 *                the label of this training sample
	 * @param wordCounts
	 *                the frequency of words (after hash trick) in the
	 *                document
	 * @param totalIters
	 *                keep track of current iteration for lazy
	 *                regularization update
	 * @param coeffLR
	 *                model parameters, to be updated
	 * @param coeffUpdateLag
	 *                lag iterations for regularization, to be updated
	 * @param vocSize
	 *                vocabulary size for the hash trick
	 * @param rate
	 *                learning rate
	 * @param regCoeff
	 *                regularization coefficients
	 */
	private static void trainOneDoc(String targetLabel, String trainLabel, Map<Integer, Integer> wordCounts,
			int totalIters, Map<Integer, Double> coeffLR, Map<Integer, Integer> coeffUpdateLag, int vocSize,
			double rate, double regCoeff) {

		// response = {0, 1} for the target label
		int response = getResponse(trainLabel, targetLabel);

		// prediction using current model parameters
		double predict = getDocPrediction(wordCounts, coeffLR, vocSize);

		// update coeffLR and coeffUpdateLag for the appeared words
		for (Integer wordID : wordCounts.keySet()) {
			int count = wordCounts.get(wordID);
			double newcoeff = 0;

			if (coeffLR.containsKey(wordID)) {
				// regularization that would have been performed
				newcoeff = coeffLR.get(wordID) * Math.pow(1 - 2 * rate * regCoeff,
						totalIters - coeffUpdateLag.get(wordID));
			}
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
	 *                the frequency of words (after hash trick) in the
	 *                document
	 * @param coeffLR
	 *                current model parameter
	 * @param vocSize
	 *                vocabulary size for has trick
	 * @return predicted probability in [0,1]
	 */
	private static double getDocPrediction(Map<Integer, Integer> wordCounts, Map<Integer, Double> coeffLR, int vocSize) {
		double score = 0;
		for (Integer wordID : wordCounts.keySet()) {
			int count = wordCounts.get(wordID);
			// only use words seen in training samples
			if (coeffLR.containsKey(wordID)) {
				score += coeffLR.get(wordID) * count;
			}
		}
		return sigmoid(score);
	}

	/**
	 * Predict labels on testing data in the testFile. Output is written to
	 * stdout.
	 * 
	 * @param targetLabels
	 *                names of all binary classifiers
	 * @param coeffLRs
	 *                trained coefficients for logistic regression, one
	 *                classifier for each label
	 * @param testFile
	 *                file name of the test data
	 * @param vocSize
	 *                the vocabulary size for hash trick
	 */
	public static void predictLR(String[] targetLabels, List<Map<Integer, Double>> coeffLRs, String testFile,
			int vocSize) throws IOException {
		int labelNumber = targetLabels.length;

		// Streaming through testing data from file
		BufferedReader testDataIn = new BufferedReader(new FileReader(testFile));
		String testDoc;

		while ((testDoc = testDataIn.readLine()) != null) {

			Vector<String> tokens = tokenizeDoc(testDoc);
			Map<Integer, Integer> wordCounts = getWordCounts(tokens, vocSize, 2);
			
			// debugging
			double maxpredict = 0;
			String predLabel = "";
			
			// prediction of probability for each label
			for (int i = 0; i < labelNumber; i++) {
				double predict = getDocPrediction(wordCounts, coeffLRs.get(i), vocSize);
				// output: label_1 probability_1, ...
				String output = String.format("%s\t%f", targetLabels[i], predict);
				if (i < labelNumber) {
					output += ",";
				}
				System.out.print(output);

				// for debugging
				if (predict > maxpredict) {
					maxpredict = predict;
					predLabel = targetLabels[i];
				}
			}

			// for debugging
			System.out.print(String.format(" || true=%s,predict=%s", tokens.elementAt(1), predLabel));

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

	/**
	 * Obtain the word frequencies in a given document.
	 * 
	 * @param tokens
	 *                vectors of strings of words
	 * @param vocSize
	 *                vocabulary size for hash trick
	 * @param startIndex
	 *                start counting from the index (inclusive)
	 * @return <key=wordID, value=count>
	 */
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
		int response = labelString.contains(targetLabel) ? 1 : 0;
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