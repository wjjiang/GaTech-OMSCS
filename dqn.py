from collections import deque
import gym
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.optimizers import Adam
import numpy as np
import random


class DQN:
	def __init__(self, env):
		## environment
		self.env = env
		## add trained trails into memory, for random sampling
		self.memory = deque(maxlen=2000)
		## model hyper-parameters
		### future rewards depreciation factor
		self.gamma = 0.95
		### firstly expore all options, then gradually shift over to exploiting
		#### fraction of time dedicated to exploring (take random action)
		#### set to 100% at first
		self.epsilon = 1.0
		#### minimal epsilon the model reaches
		self.epsilon_min = 0.01
		#### decay speed of epsilon, for every successive time step
		self.epsilon_decay = 0.995
		### standard learning rate parameter
		self.learning_rate = 0.01
		self.tau = 0.05
		## 2 models
		### do the actual predictions on what action to take
		self.model = self.create_model()
		### track what action we want our model to take
		### used for faster convergence
		self.target_model = self.create_model()

	def create_model(self):
		model = Sequential()
		state_shape = self.env.observation_space.shape
		## 1st layer maps to state dimension
		model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
		## 2 hidden layers
		model.add(Dense(48, activation="relu"))
		model.add(Dense(24, activation="relu"))
		## last layer maps to action dimension
		model.add(Dense(self.env.action_space.n))
		model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
		return model

	# DQN Training
	## Step 1: Remembering
	## add (s, a, r, s') into memory
	## also need to add 'done' for updating the reward function later
	def remember(self, state, action, reward, new_state, done):
		self.memory.append([state, action, reward, new_state, done])

	## Step 2: Learning / Replaying
	def replay(self):
		### standard batch size, suggested by Mnih et al (2013)
		batch_size = 32
		#### corner case
		if (len(self.memory) < batch_size):
			return
		else:
			#### sample 32 stored (s, a, r, s') for re-training
			samples = random.sample(self.memory, batch_size)
			#### loop each sample
			for sample in samples:
				state, action, reward, new_state, done = sample
				##### update target model
				target = self.target_model.predict(state)
				if done:
					target[0][action] = reward
				else:
					Q_future = max(self.target_model.predict(new_state)[0])
					target[0][action] = reward + Q_future*self.gamma
				##### train the Q network
				self.model.fit(state, target, epochs=1, verbose=0)

	## Step 3: re-orient goal
	## copy over weights from main model into the target model
	def target_train(self):
		### get weights from main model
		weights = self.model.get_weights()
		### get weights from target model
		target_weights = self.target_model.get_weights()
		### copy over into target model
		for i in range(len(target_weights)):
			target_weights[i] = weights[i]
		### set target weights
		self.target_model.set_weights(target_weights)

	# DQN Action
	def act(self, state):
		## decay epsilon
		self.epsilon *= self.epsilon_decay
		## floor epsilon to epsilon_min
		self.epsilon = max(self.epsilon_min, self.epsilon)
		## perform desired action based on probability
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		else:
			return np.argmax(self.model.predict(state)[0])

# Training Agent
if __name__ == "__main__":
	## set environment
	#env = gym.make('MountainCar-v0')
	env = gym.make('LunarLander-v2')
	## set # of trails, and trail length
	trails = 1000
	trail_len = 500
	## set DQN agent
	dqn_agent = DQN(env = env)
	## loop each trail
	for trail in range(trails):
		### observe initial state
		cur_state = env.reset().reshape(1, env.observation_space.shape[0])
		### loop each step until maximum allowed step reached
		for step in range(trail_len):
			#### select an action a
			action = dqn_agent.act(cur_state)
			#### turn on render
			#env.render()
			#### update (s', r)
			new_state, reward, done, _ = env.step(action)
			#### update reward
			#reward = reward if not done else -20
			#print(reward)
			#### reshape new_state
			new_state = new_state.reshape(1, env.observation_space.shape[0])
			#### remember
			dqn_agent.remember(cur_state, action, reward, new_state, done)
			#### replay
			dqn_agent.replay()
			#### re-orient goals
			dqn_agent.target_train()
			#### update s with s'
			cur_state = new_state
			#### end step-loop if done
			if done:
				break
		### report
		print("trail {}/{}, reward={}".format(trail, trails, reward))