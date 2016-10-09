#!/user/bin/bash

#######################
##
## On-line stochastic gradient descent
## for L2-regularized logistic regression
## using sparsified update
##
##############################
##
## Training and testing data format:
## one document per line:
##
## docID label1,label2,...,labelK word1 word2 ... wordN
##
## Names for the labels are hard-coded in LR.java
## Multiple binary classifiers are trained, one for each label
##############################

## path to training and testing data
TRAIN=../data/abstract.small.train
TEST=../data/abstract.small.test
## number of training samples
SIZE=44925 
## iteration for SGD
ITER=20 
## coeff for L2 regularization
REGCOEFF=0.1
## vocabulary size for hash trick
DICSIZE=10000

## run
javac LR.java

for((i=1;i<=40;i++));
do /usr/local/bin/gshuf $TRAIN;
done | java -Xmx128m LR $DICSIZE 0.5 $REGCOEFF $ITER $SIZE $TEST > prediction.txt

