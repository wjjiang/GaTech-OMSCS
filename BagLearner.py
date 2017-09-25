import numpy as np

class BagLearner(object):
    def __init__(self, learner, kwargs, bags = 20, boost =False, verbose = False):
        self.learners = []
        self.bags = bags
        # Place learners into bags
        # kwargs: keyword arguments
        for i in range(0, self.bags):
            self.learners.append(learner(**kwargs))
            
    def author(self):
        return 'wjiang84' # replace with your Georgia Tech username     
    
    def addEvidence(self,dataX,dataY):
        # Number of rows from X values of data
        nRow = dataX.shape[0]
        # Loop for each bag
        for bag in xrange(self.bags):
            ## Choose 60% rows randomly
            rowIndex = np.random.choice(range(nRow-1), int(nRow*0.6))
            ## Add the randomly chosen rows above into learner
            self.learners[bag].addEvidence(dataX[rowIndex,], dataY[rowIndex])
            
    def query(self,Xtest):
        result = 0.
        for bag in xrange(self.bags):
            result += self.learners[bag].query(Xtest)
        return result / self.bags