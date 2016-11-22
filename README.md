# RLR: Regularized Logistic Regression

This is an assginment for CMU 10-605 ["Machine Learning with Large Datasets"](http://curtis.ml.cmu.edu/w/courses/index.php/Machine_Learning_with_Large_Datasets_10-605_in_Fall_2016).
It contains a Java implementation for L2-regularized logistic regression learning with scalable on-line stochastic gradient descent. Efficient sparse updates are achieved by lazy update of regularization. The hashing trick is used for memory saving. 

The data are articles from DBPedia, and the label is the type of the article. There are in total 17 classes in the dataset, and they are from the first level class in DBpedia ontology. Each document may belong to multiple classes, and we train a separate binary classifier for each class. The data contains one document per line of the format: 

> docID    label1,label2,...    word1 word2 word3...

Given the path to testing dataset, `LR.java` streams through training data from stdin (`System.in`), and produces output in the following format, one line per test sample:

> label1  probability_label1,label2 probability_label2,...

See `run.sh` for an example of training the logistic regression using 20 iterations, and producing prediction for testing data.

