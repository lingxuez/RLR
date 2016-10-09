#!/user/bin/bash

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

javac LR.java

for((i=1;i<=40;i++));
do /usr/local/bin/gshuf $TRAIN;
done | java -Xmx128m LR $DICSIZE 0.5 $REGCOEFF $ITER $SIZE $TEST > prediction.txt