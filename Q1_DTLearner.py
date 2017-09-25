"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import DTLearner as dtl
import sys

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
    #print data

    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print("testX has shape of {}".format(testX.shape))
    print("testY has shape of {}".format(testY.shape))

    # create a decision tree learner and train it
    RMSE_DTL_InSample = []
    RMSE_DTL_OutOfSample = []
    Leaf_Sizes = range(1, 51, 1)
    for leafSize in Leaf_Sizes:
    	## Step 1: create a DTLearner
    	learner_dtl = dtl.DTLearner(leaf_size=leafSize, verbose=True)
    	## Step 2: train it
    	learner_dtl.addEvidence(trainX, trainY)
    	## Step 3: evaluate in sample
    	#### get the predictions
    	predY_dtl_insample = learner_dtl.query(trainX)
    	#### calculate RMSE
    	rmse_dtl_insample = math.sqrt(((trainY - predY_dtl_insample) ** 2).sum()/trainY.shape[0])
    	#### append to RMSE_DTL_InSample
    	RMSE_DTL_InSample.append(rmse_dtl_insample)
    	## Step 4: evaluate out of sample
    	#### get the predictions
    	predY_dtl_outofsample = learner_dtl.query(testX)
    	#### calculate RMSE
    	rmse_dtl_outofsample = math.sqrt(((testY - predY_dtl_outofsample) ** 2).sum()/testY.shape[0])
    	#### append to RMSE_DTL_OutOfSample
    	RMSE_DTL_OutOfSample.append(rmse_dtl_outofsample)

    np.savetxt('DT_LeafSize.csv', (Leaf_Sizes, RMSE_DTL_InSample, RMSE_DTL_OutOfSample), delimiter=',')