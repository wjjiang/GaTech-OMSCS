import numpy as np

class BagLearner(object):
    def __init__(self, learner, kwargs, bags = 20, boost =False, verbose = False):
        self.learners = []
        self.bags = bags
        for i in range(0, self.bags):
            self.learners.append(learner(**kwargs))
            
    def author(self):
        return 'wwan9' # replace tb34 with your Georgia Tech username     
    
    def addEvidence(self,dataX,dataY):
        nrow = dataX.shape[0]
        for i in range(0, self.bags):
            index = np.random.choice(range(nrow-1), int(nrow*0.5))
            self.learners[i].addEvidence(dataX[index,], dataY[index])
            
    def query(self,Xtest):
        res = np.zeros(Xtest.shape[0])
        for i in range(0, self.bags):
            res += self.learners[i].query(Xtest)
        return res / self.bags
