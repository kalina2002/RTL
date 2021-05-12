#!/usr/bin/env python
import gym
import numpy as np
from collections import deque
from keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense, Reshape, Flatten, Dropout
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
import random


class DQNAgent():
    def __init__(self, env_id, path='.', episodes=10, max_env_steps=500, max_env_length=8, win_threshold=0.5, epsilon_decay=0.99993,
             state_size=None, action_size=None, epsilon=1, epsilon_min=0.01, 
             gamma=1, alpha=.01, alpha_decay=.01, batch_size=16, prints=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make(env_id)

        if state_size is None:
            obs = self.env._get_screen()
            self.state_size = obs.shape
        else: 
            self.state_size = state_size

        if action_size is None: 
            self.action_size = self.env.action_space.n 
        else: 
            self.action_size = action_size

        self.episodes = episodes
        self.max_episode_steps = max_env_steps
        self.max_env_length=  max_env_length
        self.win_threshold = win_threshold
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        self.path = path                     #location where the model is saved to
        self.prints = prints                 #if true, the agent will print his scores

        self.model = self._build_model()


    def _build_model(self):
        model = Sequential()
        #model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        #model.add(Dense(48, activation='tanh'))
        #model.add(Dense(self.action_size, activation='linear'))
        #model.compile(loss='mse',
        #optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        model.add(Conv2D(32, (3, 3), input_shape=self.state_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(24))
        model.add(Activation('relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(48))
        model.add(Activation('relu'))
        model.add(Dense(96))
        model.add(Activation('relu'))
        model.add(Dense(192))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        model.add(Activation('linear'))

        model.compile(loss='mse',
            optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        return model

    def reset(self):
        return self.env.reset()

    
    def act(self, state):
        r = np.random.random()
        if (r < self.epsilon):
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state.reshape((1,) + state.shape)))

    def step(self, action):
        return self.env.step(action)
    
    def remember(self, state, action, reward, next_state, done): 
        #print('state',state)
        #print('action',action)
        #print('next state', next_state)
        self.memory.append((state, action, reward, next_state, done))
        
	#flip-lr
        #state_new = np.flip(state, 1)
        #next_state_new = np.flip(next_state, 1)
        #action_new = self.flip_lr(action)
        #self.memory.append((state_new, action_new, reward, next_state_new, done))
        


        ##print('state difference indicies', (state-next_state)[33:164,11:137,:].nonzero())
	
	#flip-up
        #if(np.array_equal(state[33:164,11:137,:],next_state[33:164,11:137,:])):
        #    print('state is equal to the next state')
	
	#    state_new = np.flip(state[33:164,11:137,:], 0)
	#    next_state_new = np.flip(next_state[33:164,11:137,:], 0)
	#    action_new = self.flip_up(action)
	#    self.memory.append((state_new, action_new, reward, next_state_new, done))



    def flip_lr(self,action):
        direction = action%4
        if direction ==1:
            new_direction = 2
        elif direction ==2:
            new_direction = 1
        else:
            new_direction = direction
        y = int(action/(9*4))
        #print('y:',y)
        x =  8 - int(int(action%(9*4))/4)
        #print('x:,',x)
        new_action = y*(9*4)+x*4+ new_direction
        
        #print('action, new action',action, new_action)
        return int(new_action)

    def flip_up(self, action):
	direction = action%4
	if direction ==0:
	    new_direction = 3
	elif direction ==3:
	    new_direction = 0
	else:
	    new_direction = direction
	
	y = 8-int(action/(9*4))
	x = int(int(action%(9*4))/4)
 	new_action = y*(9*4)+x*4+new_direction
	return int(new_action)




    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
        self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            state = state.reshape((1,) + state.shape)
            #print('training state', state.sum())
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state.reshape((1,) + next_state.shape))[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def _get_current_coverage(self):
        print('get current coverage')
        return self.env._get_current_coverage()
