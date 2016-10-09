import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

/*
 * 10-605 hw3
 * @author Lingxue Zhu
 * @version 2016/10/06
 */

/**
 * 10-605 hw3 Regularized Logistic Regression using stochastic gradient descent
 * and efficient lazy-update of regularization.
 * 
 * @author Lingxue Zhu
 * @version 2016/10/08
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
		double initRate = Double.valueOf(args[1]);
		double regCoeff = Double.valueOf(args[2]);
		int maxIter = Integer.valueOf(args[3]);
		int trainSize = Integer.valueOf(args[4]);
		String testFile = args[5];

		// 17 binary classifiers, one for each label
		String[] targetLabels = { "CelestialBody", "Biomolecule", "Organisation", "Work", "Agent", "Event",
				"Person", "ChemicalSubstance", "Place", "Location", "SportsSeason", "Activity",
				"Device", "TimePeriod", "MeanOfTransportation", "Species", "other" };

		// training; one binary classifier for each label
		Map<Integer, double[]> coeffLRs = trainLR(targetLabels, vocSize, initRate, regCoeff, maxIter,
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
	public static Map<Integer, double[]> trainLR(String[] targetLabels, int vocSize, double initRate,
			double regCoeff, int maxIter, int trainSize) throws IOException {
		// <wordID, coefficients for all classifiers>
		Map<Integer, double[]> coeffLRs = new HashMap<>();
		// <wordID, lag_iters>
		Map<Integer, int[]> coeffUpdateLags = new HashMap<>();
		// bias term (without regularization); all other wordIDs are
		// positive
		coeffLRs.put(-1, new double[targetLabels.length]);

		// Streaming through training data from stdin
		BufferedReader trainDataIn = new BufferedReader(new InputStreamReader(System.in));
		int totalIters = 0;

		for (int iter = 1; iter <= maxIter; iter++) {
			double rate = initRate / (iter * iter); // learning rate

			// One pass through training samples
			for (int n = 0; n < trainSize; n++) {
				totalIters += 1;
				String trainDoc = trainDataIn.readLine();

				// binary responses and features
				String label = trainDoc.split("\\s+")[1];
				int[] responses = getResponses(label, targetLabels);
				Map<Integer, Integer> wordCounts = getWordCounts(trainDoc, vocSize, 2);

				// update all binary classifiers
				trainOneDoc(targetLabels, responses, wordCounts, totalIters, coeffLRs, coeffUpdateLags,
						rate, regCoeff);
			}

			// lazy update for regularization
			updateRegularization(coeffLRs, coeffUpdateLags, regCoeff, rate, totalIters);

			// compute log likelihood
//			double lcl = getLogLikelihood(trainDataIn, trainSize, vocSize, targetLabels, coeffLRs);
//			System.out.println(String.format("%d\t%f", iter, lcl));
		}
		trainDataIn.close();

		return coeffLRs;
	}

	/**
	 * Compute log likelihood on given data.
	 * 
	 * @param dataIn
	 *                data input
	 * @param trainSize
	 *                number of samples
	 * @param vocSize
	 *                vocabulary size for hash trick
	 * @param targetLabels
	 *                labels to be predicted
	 * @param coeffLRs
	 *                model coefficients
	 * @return the log likelihood
	 * @throws IOException
	 */
	private static double getLogLikelihood(BufferedReader dataIn, int trainSize, int vocSize, String[] targetLabels,
			Map<Integer, double[]> coeffLRs) throws IOException {
		double lcl = 0;
		int labelNumber = targetLabels.length;

		for (int n = 0; n < trainSize; n++) {
			String trainDoc = dataIn.readLine();

			// binary responses and features
			String label = trainDoc.split("\\s+")[1];
			int[] responses = getResponses(label, targetLabels);
			Map<Integer, Integer> wordCounts = getWordCounts(trainDoc, vocSize, 2);

			// predicted probabilities
			double[] predictions = getDocPredictions(wordCounts, coeffLRs, labelNumber);

			// log likelihood
			for (int i = 0; i < labelNumber; i++) {
				lcl += responses[i] * Math.log(predictions[i]);
				lcl += (1 - responses[i]) * Math.log(1 - predictions[i]);
			}
		}

		return lcl;
	}

	/**
	 * In the end of each epoch, perform all lagged updates for
	 * regularizations.
	 * 
	 * @param coeffLRs
	 *                model coefficients
	 * @param coeffUpdateLags
	 *                lagged iters
	 * @param regCoeff
	 *                regularization coefficients
	 * @param rate
	 *                learning rate
	 * @param totalIters
	 *                current iter number
	 */
	private static void updateRegularization(Map<Integer, double[]> coeffLRs, Map<Integer, int[]> coeffUpdateLags,
			double regCoeff, double rate, int totalIters) {

		for (Integer wordID : coeffLRs.keySet()) {
			// skip bias term
			if (wordID < 0) {
				continue;
			}

			double[] newcoeff = coeffLRs.get(wordID);
			int[] updateLags = coeffUpdateLags.get(wordID);

			for (int i = 0; i < newcoeff.length; i++) {
				// regularization that would have been performed
				newcoeff[i] *= Math.pow(1 - 2 * rate * regCoeff, totalIters - updateLags[i]);
				// new lagged iterations
				updateLags[i] = totalIters;
			}

			coeffLRs.put(wordID, newcoeff);
			coeffUpdateLags.put(wordID, updateLags);
		}
	}

	/**
	 * Update the coefficients for a given classifier using gradient descent
	 * from one document.
	 * 
	 * @param targetLabels
	 *                all labels
	 * @param responses
	 *                binary response for each label for the document
	 * @param wordCounts
	 *                the frequency of words (after hash trick) in the
	 *                document
	 * @param totalIters
	 *                keep track of current iteration for lazy
	 *                regularization update
	 * @param coeffLRs
	 *                model parameters, to be updated
	 * @param coeffUpdateLags
	 *                lag iterations for regularization, to be updated
	 * @param rate
	 *                learning rate
	 * @param regCoeff
	 *                regularization coefficients
	 */
	private static void trainOneDoc(String[] targetLabels, int[] responses, Map<Integer, Integer> wordCounts,
			int totalIters, Map<Integer, double[]> coeffLRs, Map<Integer, int[]> coeffUpdateLags,
			double rate, double regCoeff) {
		int labelNumber = targetLabels.length;

		// predicted probabilities using current model parameters
		double[] predict = getDocPredictions(wordCounts, coeffLRs, labelNumber);

		// update bias term
		double[] bias = coeffLRs.get(-1);
		for (int i = 0; i < labelNumber; i++) {
			bias[i] += rate * (responses[i] - predict[i]);
		}
		coeffLRs.put(-1, bias);

		// update coeffLRs and coeffUpdateLags for the appeared words
		for (Integer wordID : wordCounts.keySet()) {
			int count = wordCounts.get(wordID);
			// initialize at 0
			double[] newcoeff = new double[labelNumber];
			int[] updateLags = new int[labelNumber];

			// regularization that would have been performed
			if (coeffLRs.containsKey(wordID)) {
				newcoeff = coeffLRs.get(wordID);
				updateLags = coeffUpdateLags.get(wordID);
				for (int i = 0; i < labelNumber; i++) {
					newcoeff[i] *= Math.pow(1 - 2 * rate * regCoeff, totalIters - updateLags[i]);
				}
			}
			for (int i = 0; i < labelNumber; i++) {
				// gradient descent contributed from data
				newcoeff[i] += rate * (responses[i] - predict[i]) * count;
				// new lag iterations
				updateLags[i] = totalIters;
			}

			// update
			coeffLRs.put(wordID, newcoeff);
			coeffUpdateLags.put(wordID, updateLags);
		}
	}

	/**
	 * Get prediction of probability for one example using current parameter
	 * for logistic regression.
	 * 
	 * @param wordCounts
	 *                the frequency of words (after hash trick) in the
	 *                document
	 * @param coeffLRs
	 *                current model parameter
	 * @param labelNumber
	 *                number of classis
	 * @return predicted probability in [0,1] for each classifier, length is
	 *         labelNumber
	 */
	private static double[] getDocPredictions(Map<Integer, Integer> wordCounts, Map<Integer, double[]> coeffLRs,
			int labelNumber) {
		// bias term
		double[] score = coeffLRs.get(-1);

		// features: words in document
		for (Integer wordID : wordCounts.keySet()) {
			int count = wordCounts.get(wordID);
			// only use words seen in training samples
			if (coeffLRs.containsKey(wordID)) {
				double[] coeff = coeffLRs.get(wordID);
				for (int i = 0; i < labelNumber; i++) {
					score[i] += coeff[i] * count;
				}
			}
		}

		// convert to probability
		for (int i = 0; i < labelNumber; i++) {
			score[i] = sigmoid(score[i]);
		}
		return score;
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
	public static void predictLR(String[] targetLabels, Map<Integer, double[]> coeffLRs, String testFile,
			int vocSize) throws IOException {
		int labelNumber = targetLabels.length;

		// -- START for accuracy --
		// double correctNumber = 0, totalDoc = 0;
		// -- END for accuracy --

		// Streaming through testing data from file
		BufferedReader testDataIn = new BufferedReader(new FileReader(testFile));
		String testDoc;
		while ((testDoc = testDataIn.readLine()) != null) {
			Map<Integer, Integer> wordCounts = getWordCounts(testDoc, vocSize, 2);

			// prediction of probability for each label
			double[] predictions = getDocPredictions(wordCounts, coeffLRs, labelNumber);

			// print out all pairs of <label, predicted probability>
			for (int i = 0; i < labelNumber - 1; i++) {
				System.out.print(String.format("%s\t%f,", targetLabels[i], predictions[i]));
			}
			// last label: change line
			System.out.println(String.format("%s\t%f", targetLabels[labelNumber - 1],
					predictions[labelNumber - 1]));

			// -- START for accuracy: correctly classified numbers
			// --
			// String label = testDoc.split("\\s+")[1];
			// int[] trueResponses = getResponses(label,
			// targetLabels);
			// for (int i = 0; i < labelNumber; i++) {
			// if ( (predictions[i] > 0.5 && trueResponses[i] == 1)
			// ||
			// (predictions[i] < 0.5 && trueResponses[i] == 0)) {
			// correctNumber++;
			// }
			// }
			// totalDoc++;
			// -- END for accuracy --
		}
		testDataIn.close();

		// -- START for accuracy --
		// System.out.println(String.format("Accuracy=%.4f",
		// 100*correctNumber / (totalDoc * labelNumber)));
		// -- END for accuracy --
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
	 * Tokenize document, and obtain the word frequencies.
	 * 
	 * @param curDoc
	 *                string for the full text in document
	 * @param vocSize
	 *                vocabulary size for hash trick
	 * @param startIndex
	 *                words start from this index (inclusive)
	 * @return <key=wordID, value=count>
	 */
	private static Map<Integer, Integer> getWordCounts(String curDoc, int vocSize, int startIndex) {
		// tokenize
		String[] words = curDoc.split("\\s+");

		// get counts, starting from given index
		Map<Integer, Integer> wordCounts = new HashMap<>();
		for (int i = startIndex; i < words.length; i++) {
			// tokenize
			words[i] = words[i].replaceAll("\\W", "").toLowerCase();
			// counting
			int wordID = wordToID(words[i], vocSize);
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
	 *                the label(s) of a document, multiple labels are
	 *                separated by ","
	 * @param targetLabels
	 *                the target labels for the binary classifiers
	 * @return responses {0,1} indicating whether document belongs to each
	 *         class
	 * 
	 */
	private static int[] getResponses(String labelString, String[] targetLabels) {
		int[] responses = new int[targetLabels.length];
		for (int i = 0; i < targetLabels.length; i++) {
			responses[i] = labelString.contains(targetLabels[i]) ? 1 : 0;
		}
		return responses;
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