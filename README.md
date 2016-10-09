# Regularized_Logistic_Regression

Learn an L2-regularized logistic regression using on-line stochastic gradient descent. Efficient sparse updates are achieved by lazy update of regularization. The hashing trick is used for memory saving. 

The data are articles from DBPedia, and the label is the type of the article. There are in total 17 classes in the dataset, and they are from the first level class in DBpedia ontology. Each document may belong to multiple classes, and we train a separate binary classifier for each class. The data contains one document per line of the format: 

> docID    label1,label2,...    word1 word2 word3...

Given the path to testing dataset, `LR.java` streams through training data from stdin (`System.in`), and produces output in the following format, one line per test sample:

> label1  probability_label1,label2 probability_label2,...

See `run.sh` for an example of training the logistic regression using 20 iterations, and producing prediction for testing data.

