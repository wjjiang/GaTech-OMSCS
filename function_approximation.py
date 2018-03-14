# =============================================================================
# function approximators to effectively 
# map continuous observations to discrete states
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tempfile
import base64
import pprint
import random
import json
import sys
import gym
import io

from gym import wrappers
from collections import deque
from subprocess import check_output
from IPython.display import HTML

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# select actions
def action_selection(state, model, episode, n_episodes):
    epsilon = 0.99 if episode < n_episodes//4 else 0.33 if episode < n_episodes//2 else 0.
    values = model.predict(state.reshape(1, 4))[0]
    if np.random.random() < epsilon:
        action = np.random.randint(len(values))
    else:
        action = np.argmax(values)
    return action, epsilon

# Q-Learning algorithm
def neuro_q_learning(env, gamma = 0.99):
    nS = env.observation_space.shape[0]
    nA = env.env.action_space.n
    
    # memory bank, known as 'experience replay' for DQN
    memory_bank = deque()
    memory_bank_size = 100000
    
    # function approximator
    model = Sequential()
    model.add(Dense(64, input_dim=nS, activation='relu'))
    model.add(Dense(nA, activation='linear'))
    model.compile(loss='mse', optimizer='adam')

    # constant values
# =============================================================================
#     n_episodes = 50000
#     batch_size = 256
#     training_frequency = 20
# =============================================================================
    n_episodes = 1000
    batch_size = 32
    training_frequency = 4
    
    # for statistics
    epsilons = []
    states = []
    actions = []
    
    # interactions
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        
        # each episode
        while not done:
            states.append(state)
            
            # select action
            action, epsilon = action_selection(state, model, episode, n_episodes)
            epsilons.append(epsilon)
            actions.append(action)
            
            # save history in memory bank
            nstate, reward, done, info = env.step(action)
            memory_bank.append((state, action, reward, nstate, done))
            if len(memory_bank) > memory_bank_size:
                memory_bank.popleft()
            
            # iterate to next state
            state = nstate

        # only every few episodes enter training and update neural network weights
        if episode % training_frequency == 0 and len(memory_bank) == memory_bank_size:
            
            # randomly select batches of samples from the history
            # for training to prevent values spiking due to high 
            # correlation of sequential values
            minibatch = np.array(random.sample(memory_bank, batch_size))

            # extract values by type from the minibatch
            state_batch = np.array(minibatch[:,0].tolist())
            action_batch = np.array(minibatch[:,1].tolist())
            rewards_batch = np.array(minibatch[:,2].tolist())
            state_prime_batch = np.array(minibatch[:,3].tolist())
            is_terminal_batch = np.array(minibatch[:,4].tolist())

            # use the current neural network to predict 
            # current state values and next state values
            state_value_batch = model.predict(state_batch)
            next_state_value_batch = model.predict(state_prime_batch)

            # update the state values given the batch
            for i in range(len(minibatch)):
                if is_terminal_batch[i]:
                    state_value_batch[i, action_batch[i]] = rewards_batch[i]
                else:
                    state_value_batch[i, action_batch[i]] = rewards_batch[i] + gamma * np.max(next_state_value_batch[i])
            
            # update the neural network weights
            model.train_on_batch(state_batch, state_value_batch)

    return model, (epsilons, states, actions)


# training
env = gym.make('CartPole-v0')
model, stats = neuro_q_learning(env)
# close environment
env.close()

# statistical analysis
epsilons, states, actions = stats
## plot epsilons
#plt.plot(np.arange(len(epsilons)), epsilons, '.')
#plt.savefig('epsilons.png')
## plot histogram
hist, bins = np.histogram(actions, bins=3)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.savefig('actions_histogram.png')

# testing
env = gym.make('CartPole-v0')
for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(model.predict(state.reshape(1, 4))[0])
        nstate, reward, done, info = env.step(action)
        state = nstate
env.close()
