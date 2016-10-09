#!/user/bin/bash

##############################
## Lingxue Zhu
## 2016/10/09
##############################
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
##
##############################

## path to training and testing data
TRAIN=../data/abstract.small.train
TEST=../data/abstract.small.test
## number of training samples, i.e., wc -l $TRAIN
SIZE=44925 
## iteration to perform for SGD
ITER=20 
## coefficient for L2 regularization
REGCOEFF=0.1
## vocabulary size for hashing trick
DICSIZE=10000

## run
javac LR.java

for((i = 1; i <= $ITER; i++));
do /usr/local/bin/gshuf $TRAIN; ## randomly shuffle data
done | java -Xmx128m LR $DICSIZE 0.5 $REGCOEFF $ITER $SIZE $TEST > prediction.txt

