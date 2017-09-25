import LinRegLearner as lrl
import BagLearner as bl

class InsaneLearner(object):
    def __init__(self, verbose = False):
        self.learners = []
        for bag in xrange(20):
            self.learners.append(bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False))

    def author(self):
        return 'wjiang84' # replace with your Georgia Tech username

    def addEvidence(self,dataX,dataY):
        for bag in xrange(20):
            self.learners[bag].addEvidence(dataX, dataY)

    def query(self,Xtest):
        result = sum(self.learners[bag].query(Xtest) for bag in xrange(20))
        return result / 20.