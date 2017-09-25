"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import DTLearner as dtl
import RTLearner as rtl
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

    # # Step 1: create a linear regression learner and train it
    # ## Step 1.1: create a LinRegLearner
    # learner_lrl = lrl.LinRegLearner(verbose = True)
    # ## Step 1.2: train it
    # learner_lrl.addEvidence(trainX, trainY)
    # print("Author of this learning is {}".format(learner_lrl.author()))
    # ## Step 1.3: evaluate in sample
    # predY_lrl = learner_lrl.query(trainX) # get the predictions
    # rmse_lrl = math.sqrt(((trainY - predY_lrl) ** 2).sum()/trainY.shape[0])
    # print
    # print "Linear Regression Learner"
    # print "In sample results"
    # print("RMSE: {}".format(rmse_lrl))
    # c_lrl = np.corrcoef(predY_lrl, y=trainY)
    # print("corr: {}".format(c_lrl[0,1]))
    # # Step 1.4: evaluate out of sample
    # predY_lrl = learner_lrl.query(testX) # get the predictions
    # rmse_lrl = math.sqrt(((testY - predY_lrl) ** 2).sum()/testY.shape[0])
    # print "Out of sample results"
    # print("RMSE: {}".format(rmse_lrl))
    # c_lrl = np.corrcoef(predY_lrl, y=testY)
    # print("corr: {}".format(c_lrl[0,1]))

    # # Step 2: create a random tree learner and train it
    # ## Step 2.1: create a RTLearner
    # learner_rtl = rtl.RTLearner(verbose = True)
    # ## Step 2.2: train it
    # learner_rtl.addEvidence(trainX, trainY)
    # ## Step 2.3: evaluate in sample
    # predY_rtl = learner_rtl.query(trainX) # get the predictions
    # rmse_rtl = math.sqrt(((trainY - predY_rtl) ** 2).sum()/trainY.shape[0])
    # print
    # print "Random Tree Learner"
    # print "In sample results"
    # print("RMSE: {}".format(rmse_rtl))
    # c_rtl = np.corrcoef(predY_rtl, y=trainY)
    # print("corr: {}".format(c_rtl[0,1]))
    # # Step 2.4: evaluate out of sample
    # predY_rtl = learner_rtl.query(testX) # get the predictions
    # rmse_rtl = math.sqrt(((testY - predY_rtl) ** 2).sum()/testY.shape[0])
    # print "Out of sample results"
    # print("RMSE: {}".format(rmse_rtl))
    # c_rtl = np.corrcoef(predY_rtl, y=testY)
    # print("corr: {}".format(c_rtl[0,1]))

    # Step 3: create a decision tree learner and train it
    RMSE_DTL_InSample = []
    RMSE_DTL_OutOfSample = []
    Leaf_Sizes = range(1, 51, 1)
    for leafSize in Leaf_Sizes:
    	## Step 3.1: create a DTLearner
    	learner_dtl = dtl.DTLearner(leaf_size=leafSize, verbose=True)
    	## Step 3.2: train it
    	learner_dtl.addEvidence(trainX, trainY)
    	## Step 3.3: evaluate in sample
    	#### get the predictions
    	predY_dtl_insample = learner_dtl.query(trainX)
    	#### calculate RMSE
    	rmse_dtl_insample = math.sqrt(((trainY - predY_dtl_insample) ** 2).sum()/trainY.shape[0])
    	#### append to RMSE_DTL_InSample
    	RMSE_DTL_InSample.append(rmse_dtl_insample)
    	## Step 3.4: evaluate out of sample
    	#### get the predictions
    	predY_dtl_outofsample = learner_dtl.query(testX)
    	#### calculate RMSE
    	rmse_dtl_outofsample = math.sqrt(((testY - predY_dtl_outofsample) ** 2).sum()/testY.shape[0])
    	#### append to RMSE_DTL_OutOfSample
    	RMSE_DTL_OutOfSample.append(rmse_dtl_outofsample)

    np.savetxt('DT_LeafSize.csv', (Leaf_Sizes, RMSE_DTL_InSample, RMSE_DTL_OutOfSample), delimiter=',')

    # Step 4: create a Bagged Learner with DTLearner and train it
    RMSE_BL_InSample = []
    RMSE_BL_OutOfSample = []
    Leaf_Sizes = range(1, 51, 1)
    for leafSize in Leaf_Sizes:
    	## Step 4.1: create a DTLearner
    	learner_bl = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size":leafSize}, bags=20, boost=False, verbose=False)
    	## Step 4.2: train it
    	learner_bl.addEvidence(trainX, trainY)
    	## Step 4.3: evaluate in sample
    	#### get the predictions
    	predY_bl_insample = learner_bl.query(trainX)
    	#### calculate RMSE
    	rmse_bl_insample = math.sqrt(((trainY - predY_bl_insample) ** 2).sum()/trainY.shape[0])
    	#### append to RMSE_DTL_InSample
    	RMSE_BL_InSample.append(rmse_bl_insample)
    	## Step 4.4: evaluate out of sample
    	#### get the predictions
    	predY_bl_outofsample = learner_bl.query(testX)
    	#### calculate RMSE
    	rmse_bl_outofsample = math.sqrt(((testY - predY_bl_outofsample) ** 2).sum()/testY.shape[0])
    	#### append to RMSE_DTL_OutOfSample
    	RMSE_BL_OutOfSample.append(rmse_bl_outofsample)
    	
    np.savetxt('BL_LeafSize.csv', (Leaf_Sizes, RMSE_BL_InSample, RMSE_BL_OutOfSample), delimiter=',')