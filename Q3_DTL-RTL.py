import numpy as np
import math
import DTLearner as dtl
import RTLearner as rtl
import sys

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

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

    # # Step 1: create a random tree learner and train it
    # ## Step 1.1: create a RTLearner
    # learner_rtl = rtl.RTLearner(verbose = True)
    # ## Step 1.2: train it
    # learner_rtl.addEvidence(trainX, trainY)
    # ## Step 1.3: evaluate in sample
    # predY_rtl = learner_rtl.query(trainX) # get the predictions
    # rmse_rtl = math.sqrt(((trainY - predY_rtl) ** 2).sum()/trainY.shape[0])
    # print
    # print "Random Tree Learner"
    # print "In sample results"
    # print("RMSE: {}".format(rmse_rtl))
    # c_rtl = np.corrcoef(predY_rtl, y=trainY)
    # print("corr: {}".format(c_rtl[0,1]))
    # # Step 1.4: evaluate out of sample
    # predY_rtl = learner_rtl.query(testX) # get the predictions
    # rmse_rtl = math.sqrt(((testY - predY_rtl) ** 2).sum()/testY.shape[0])
    # print "Out of sample results"
    # print("RMSE: {}".format(rmse_rtl))
    # c_rtl = np.corrcoef(predY_rtl, y=testY)
    # print("corr: {}".format(c_rtl[0,1]))

    # Step 1: create a decision tree learner and train it
    RMSE_DTL_InSample = []
    Corr_DTL_InSample = []
    RMSE_DTL_OutOfSample = []
    Corr_DTL_OutOfSample = []
    RMSE_RTL_InSample = []
    Corr_RTL_InSample = []
    RMSE_RTL_OutOfSample = []
    Corr_RTL_OutOfSample = []
    Leaf_Sizes = range(1, 51, 1)
    for leafSize in Leaf_Sizes:
    	## Step 1: create DTLearner & RTLearner
    	learner_dtl = dtl.DTLearner(leaf_size=leafSize, verbose=True)
        learner_rtl = rtl.RTLearner(leaf_size=leafSize, verbose=True)
    	## Step 2: train
    	learner_dtl.addEvidence(trainX, trainY)
        learner_rtl.addEvidence(trainX, trainY)
    	## Step 3: evaluate in sample
    	#### get the predictions
    	predY_dtl_insample = learner_dtl.query(trainX)
        predY_rtl_insample = learner_rtl.query(trainX)
    	#### calculate RMSE & append
    	rmse_dtl_insample = math.sqrt(((trainY - predY_dtl_insample) ** 2).sum()/trainY.shape[0])
        RMSE_DTL_InSample.append(rmse_dtl_insample)
        rmse_rtl_insample = math.sqrt(((trainY - predY_rtl_insample) ** 2).sum()/trainY.shape[0])
        RMSE_RTL_InSample.append(rmse_rtl_insample)
        #### calculate correlation coefficients & append
        c_dtl_insample = np.corrcoef(predY_dtl_insample, y=testY)[0,1]
        Corr_DTL_InSample.append(c_dtl_insample)
        c_rtl_insample = np.corrcoef(predY_rtl_insample, y=testY)[0,1]
        Corr_RTL_InSample.append(c_rtl_insample)
    	## Step 3.4: evaluate out of sample
    	#### get the predictions
    	predY_dtl_outofsample = learner_dtl.query(testX)
        predY_rtl_outofsample = learner_rtl.query(testX)
    	#### calculate RMSE & append
    	rmse_dtl_outofsample = math.sqrt(((testY - predY_dtl_outofsample) ** 2).sum()/testY.shape[0])
        RMSE_DTL_OutOfSample.append(rmse_dtl_outofsample)
        rmse_rtl_outofsample = math.sqrt(((testY - predY_rtl_outofsample) ** 2).sum()/testY.shape[0])
        RMSE_RTL_OutOfSample.append(rmse_rtl_outofsample)
    	#### calculate correlation coefficients & append
        c_dtl_outofsample = np.corrcoef(predY_dtl_outofsample, y=testY)[0,1]
        Corr_DTL_OutOfSample.append(c_dtl_outofsample)
        c_rtl_outofsample = np.corrcoef(predY_rtl_outofsample, y=testY)[0,1]
        Corr_RTL_OutOfSample.append(c_rtl_outofsample)

    np.savetxt('DTL-vs-RTL.csv', (Leaf_Sizes, RMSE_DTL_InSample, Corr_DTL_InSample, RMSE_DTL_OutOfSample, Corr_DTL_OutOfSample,\
        RMSE_RTL_InSample, Corr_RTL_InSample, RMSE_RTL_OutOfSample, Corr_RTL_OutOfSample), delimiter=',')