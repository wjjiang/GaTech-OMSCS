import numpy as np

class DTLearner(object):

    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size

    def author(self):
        return 'wjiang84' # replace with your Georgia Tech username 
    
    def build_tree(self, data):
        # Case 1: reach leaves -- mark as -1
        if data.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data[:,-1]), np.nan, np.nan]])
        # Case 2: y has almost the same data -- mark as -1 for leaves
        elif abs(np.max(data[:,-1])- np.min(data[:,-1])) < 1e-6:
            return np.array([[-1, np.mean(data[:,-1]), np.nan, np.nan]])
        # Case 3: otherwise, recursively build random tree
        else:
            ## Step 1: determine feature i to split on
            ## Information gain: Correlation
            covariance_matrix = np.corrcoef(data[:, 0:-1].T, data[:, -1])
            coefficient_array = covariance_matrix[data.shape[1]-1][0:-1]
            i = np.argmax(abs(coefficient_array))
            ## Step 2: sort data matrix by i-th column
            data = data[np.argsort(data[:, i])]
            ## Step 3: calculate splitVal
            split_val = np.median(data[:, i])
            ## Step 3.1: special case (all data <= median)
            if all(s <= split_val for s in data[:, i]):
                return np.array([[-1, np.mean(data[:,-1]), np.nan, np.nan]])
            ## Step 3.2: special case (all data >= median)
            elif all(s >= split_val for s in data[:, i]):
                return np.array([[-1, np.mean(data[:,-1]), np.nan, np.nan]])
            ## Step 4: recursively build left tree
            left_tree = self.build_tree(data[data[:,i] <= split_val])
            ## Step 5: recursively build right tree
            right_tree = self.build_tree(data[data[:,i] > split_val])
            ## Step 6: after building left&right trees, define root
            ## [root-i, split_val, relative-left-position-to-root-i, \
            ## relative-right-position-to-root-i]
            root = np.array([i, split_val, 1, left_tree.shape[0] + 1])
            ## Step 7: combine root with left&right trees
            return np.vstack((root, left_tree, right_tree))

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # Number of rows from X values of data to add
        nRow = dataX.shape[0]
        # Number of columns from X values of data to add
        nCol = dataX.shape[1]
        # training data to learner, combining dataX with dataY
        trainingData = np.zeros((nRow, nCol+1))
        trainingData[:, 0:nCol] = dataX
        trainingData[:, -1] = dataY
        # call build_tree() function to recursively build Random Tree
        self.para = self.build_tree(trainingData)

    def current_node_query(self, test):    
        current_node = 0
        ## Find leaves
        while (int(self.para[current_node,0]) != -1):
            ## Find the current node list
            decision_node = self.para[current_node, :]
            ## 1st column: root node
            node_id =  int(decision_node[0])
            ## 2nd column: split value
            split_val = decision_node[1]
            #next_step = 0
            ## 3rd column: left tree node relative to root node
            if test[node_id] <= split_val:
                next_step = decision_node[2]
            ## 4th column: right tree node relative to root node
            else:
                next_step = decision_node[3]
            ## Determine current node ID
            current_node += int(next_step)
        return self.para[current_node, 1]    
    
    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        # call 
        return np.apply_along_axis(lambda x: self.current_node_query(x), 1, points)
        

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
