"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):
        self.num_states = num_states
        self.num_actions = num_actions
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = 0.99
        self.dyna = dyna
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0



    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = np.argmax(Q[s_prime,:])
        action_tmp = rand.randint(0, self.num_actions-1)
        decsion = rand.random()
        if (decsion < self.rar):
            action = action_tmp
            self.rar *= self.radr
        if self.verbose: print ("s =", s ,"a =",action)
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        action = np.argmax(Q[s_prime,:])
        Q[self.s, self.a] = (1 - self.alpha) * Q[self.s, self.a] + self.alpha * (r + self.gamma * Q[s_prime, action])
        action_tmp = rand.randint(0, self.num_actions-1)
        decsion = rand.random()
        if (decsion < self.rar):
            action = action_tmp
            self.rar *= self.radr
        if self.verbose: print ("s =", s_prime,"a =",action,"r =",r)
        return action

if __name__=="__main__":
    print ("Remember Q from Star Trek? Well, this isn't him")
