"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import DTLearner as dtl
import BagLearner as bl
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

    # create a Bagged Learner with DTLearner and train it
    RMSE_BL_InSample = []
    RMSE_BL_OutOfSample = []
    Leaf_Sizes = range(1, 51, 1)
    for leafSize in Leaf_Sizes:
    	## Step 1: create a DTLearner
    	learner_bl = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size":leafSize}, bags=20, boost=False, verbose=False)
    	## Step 2: train it
    	learner_bl.addEvidence(trainX, trainY)
    	## Step 3: evaluate in sample
    	#### get the predictions
    	predY_bl_insample = learner_bl.query(trainX)
    	#### calculate RMSE
    	rmse_bl_insample = math.sqrt(((trainY - predY_bl_insample) ** 2).sum()/trainY.shape[0])
    	#### append to RMSE_DTL_InSample
    	RMSE_BL_InSample.append(rmse_bl_insample)
    	## Step 4: evaluate out of sample
    	#### get the predictions
    	predY_bl_outofsample = learner_bl.query(testX)
    	#### calculate RMSE
    	rmse_bl_outofsample = math.sqrt(((testY - predY_bl_outofsample) ** 2).sum()/testY.shape[0])
    	#### append to RMSE_DTL_OutOfSample
    	RMSE_BL_OutOfSample.append(rmse_bl_outofsample)
    	
    np.savetxt('BL_LeafSize.csv', (Leaf_Sizes, RMSE_BL_InSample, RMSE_BL_OutOfSample), delimiter=',')