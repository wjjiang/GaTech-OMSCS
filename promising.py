import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.optimizers import SGD, RMSprop, Adam, Adamax

def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()

env = gym.make('LunarLander-v2')

# build a set of samples so we can get a scaler fitted.
observation_samples = []

# play a bunch of games randomly and collect observations
for n in range(1000):
    observation = env.reset()
    observation_samples.append(observation)
    done = False
    while not done:
        action = np.random.randint(0, env.action_space.n)
        observation, reward, done, _ = env.step(action)
        observation_samples.append(observation)
        
observation_samples = np.array(observation_samples)

#env = wrappers.Monitor(env, 'monitor-folder', force=True)

# Create scaler and fit
scaler = StandardScaler()
scaler.fit(observation_samples)

# Using the excellent Keras to build standard feedforward neural network.
# single output node, linear activation on the output
# To keep things simple,  one NN is created per action.  So
# in this problem, 4 independant neural networks are create
# Admax optimizer is the most efficient one, using default parameters.

def create_nn():
    model = Sequential()
    model.add(Dense(128, init='lecun_uniform', input_shape=(8,)))
    model.add(Activation('relu'))
#     model.add(Dropout(0.3)) #I'm not using dropout, but maybe you wanna give it a try?

    model.add(Dense(256, init='lecun_uniform'))
    model.add(Activation('tanh'))
#     model.add(Dropout(0.5))

    model.add(Dense(1, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

#     rms = RMSprop(lr=0.005)
#     sgd = SGD(lr=0.1, decay=0.0, momentum=0.0, nesterov=False)
# try "adam"
#     adam = Adam(lr=0.0005)
    adamax = Adamax() #Adamax(lr=0.001)
    model.compile(loss='mse', optimizer=adamax)
#     model.summary()
    return model

# Holds one nn for each action
class Model:
  def __init__(self, env, scaler):
    self.env = env
    self.scaler = scaler
    self.models = []
    for i in range(env.action_space.n):
        model = create_nn()  # one nn per action
        self.models.append(model) 

  def predict(self, s):
    X = self.scaler.transform(np.atleast_2d(s))
    return np.array([m.predict(np.array(X), verbose=0)[0] for m in self.models])

  def update(self, s, a, G):
    X = self.scaler.transform(np.atleast_2d(s))
    self.models[a].fit(np.array(X), np.array([G]), nb_epoch=1, verbose=0)

  def sample_action(self, s, eps):
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))

def play_one(env, model, eps, gamma):
  observation = env.reset()
  done = False
  full_reward_received = False
  totalreward = 0
  iters = 0
  while not done:
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)
            
    # update the model
    # standard Q learning TD(0)
    next = model.predict(observation)
    G = reward + gamma*np.max(next)
    model.update(prev_observation, action, G)
    totalreward += reward
    iters += 1

model = Model(env, scaler)
gamma = 0.99

N = 8010
totalrewards = np.empty(N)
costs = np.empty(N)
for n in range(N):
    eps = 1.0/np.sqrt(n+1)
    totalreward, iters = play_one(env, model, eps, gamma)
    totalrewards[n] = totalreward
    if n % 100 == 0:
      print("episode:", n, "iters", iters, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())
    if totalrewards[max(0, n-100):(n+1)].mean() >= 200:
        break

print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
print("total steps:", totalrewards.sum())

plt.plot(totalrewards)
plt.title("Rewards")
plt.show()

plot_running_avg(totalrewards)

env.close()
  return totalreward, iters