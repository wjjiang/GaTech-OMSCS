import numpy as np

class RTLearner(object):

    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        #self.para
    def author(self):
        return 'wwan9' # replace tb34 with your Georgia Tech username 
    
    def build_tree(self, data):
        #print("data: ", data)
        if data.shape[0] <= self.leaf_size:
            tmp_array = np.mean(data[:,-1])
            return np.array([[-1, tmp_array, np.nan, np.nan]])
        elif abs(np.max(data[:,-1])- np.min(data[:,-1])) < 0.00001 :
            tmp_array = np.mean(data[:,-1])
            return np.array([[-1, tmp_array, np.nan, np.nan]])
        else:
            while(True):
                i = np.random.choice(data.shape[1]-1, 1)[0]
                #print("i: ", i)
                row_x = np.random.choice(data.shape[0], 1)[0]
                row_y = np.random.choice(data.shape[0], 1)[0]
                split_val = ( data[row_x,i] + data[row_y,i]) / 2
                tmp = data[data[:,i] <= split_val].shape[0]
                if (tmp != 0 and tmp != data.shape[0]): break                

            left_tree = self.build_tree(data[data[:,i] <= split_val])
            #print(left_tree)
            right_tree = self.build_tree(data[data[:,i] > split_val])
            #print(right_tree)
            root = np.array([i, split_val, 1, left_tree.shape[0] + 1])
            return np.vstack((root, left_tree, right_tree))
    

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        nrow = dataX.shape[0]
        ncol = dataX.shape[1]
        data = np.zeros((nrow, ncol+1))
        data[:,0:ncol] = dataX
        data[:, -1] = dataY
        self.para = self.build_tree(data)
    def single_query(self, test):    
        current_node = 0
        while (int(self.para[current_node,0]) != -1):
            decision_node = self.para[current_node, :]
            split_val = decision_node[1]
            node_id =  int(decision_node[0])
            next_step = 0
            if test[node_id] <= split_val:
                next_step = decision_node[2]
            else:
                next_step = decision_node[3]
            current_node += int(next_step)
        #print(current_node)
        return self.para[current_node,1]    
    
    def query(self,Xtest):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        # some random data vector
        return np.apply_along_axis(lambda x: self.single_query(x), 1, Xtest)
        

if __name__=="__main__":
    
    Xtrain = data[:, 0:7]
    Ytrain = data[:,-1]
    learner = RTLearner(leaf_size = 1, verbose = False) # constructor
    learner.addEvidence(Xtrain, Ytrain) # training step
    Y = learner.query(Xtrain) # query
