"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def author(self):
        return 'wwan9' 
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
        self.Q = np.random.uniform(-1,1, (self.num_states, self.num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = 0.99
        self.dyna = dyna
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.T = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.Tc = np.ones((self.num_states, self.num_actions, self.num_states)) * 0.00001
        self.R = np.zeros((self.num_states, self.num_actions))
        self.D = []

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = np.argmax(self.Q[s,:])
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
        action = np.argmax(self.Q[s_prime,:])
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (r + self.gamma * self.Q[s_prime, action])
        #self.Tc[self.s, self.a, s_prime] += 1
        #self.T[self.s, self.a, s_prime] = self.Tc[self.s, self.a, s_prime] / np.sum(self.Tc[self.s, self.a, :])
        #self.R[self.s, self.a] = (1-self.alpha) * self.R[self.s, self.a] + self.alpha*r
        count = self.dyna
        self.D.append((self.s, self.a, s_prime, r))
        while(count):
            #s_tmp = rand.randint(0, self.num_states - 1)
            #a_tmp = rand.randint(0, self.num_actions - 1)
            #s_next = np.argmax(self.T[s_tmp, a_tmp, :])
            #a_next = np.argmax(self.Q[s_next, :])
            #r_new = self.R[s_tmp, a_tmp]i
            index = rand.randint(0, len(self.D) - 1)
            s_tmp,a_tmp, s_next, r_new = self.D[index]
            a_next = np.argmax(self.Q[s_next, :])
            self.Q[s_tmp, a_tmp] = (1- self.alpha) * self.Q[s_tmp, a_tmp] + self.alpha * ( r_new + self.gamma * self.Q[s_next, a_next])
            count -= 1 
        action = np.argmax(self.Q[s_prime,:])
        action_tmp = rand.randint(0, self.num_actions-1)
        decsion = rand.random()
        if (decsion < self.rar):
            action = action_tmp
            self.rar *= self.radr
        if self.verbose: print ("s =", s_prime,"a =",action,"r =",r)
        self.s = s_prime
        self.a = action
        return action

if __name__=="__main__":
    print ("Remember Q from Star Trek? Well, this isn't him")

