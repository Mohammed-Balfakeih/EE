from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import gym
from gym import spaces
#THINK I'M JUST GONNA USE POLICY GRADIENTS AAAA
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, action_space, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.action_space = action_space
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, action_space.shape[0]), dtype=dtype) #???
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

def build_dqn(lr, action_space, input_dims, fc1_dims, fc2_dims): #learning rate, fully connected layer dimensions
    model = Sequential([
        Dense(fc1_dims, input_shape = (input_dims,)), #Dense layer is connected to all previous neurons
        Activation('relu'),
        Dense(fc2_dims),
        Activation('relu'),
        Dense(action_space.shape[0])])
    model.compile(optimizer = Adam(lr=lr), loss='mse')

    return model

class Agent(object):
    def __init__(self, alpha, gamma, epsilon, batch_size, action_space, 
                input_dims, epsilon_dec=0.996, epsilon_end=0.01,
                mem_size=1000000, fname='dqn_model.h5'): #alpha = learning rate, gamma = discount rate, epsilon for epsilon greedy, epsilon_dec to decrease epsilon over time, epsilon_end is min
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.fname = fname

        self.memory = ReplayBuffer(mem_size, input_dims, self.action_space, discrete=False) 
 
        self.q_eval = build_dqn(alpha, self.action_space, input_dims, 256, 256) #256, 256 is default size, this is building a network to evaluate the actions

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon: #epsilon greedy algorithm
            action = action = np.random.uniform(-1.0, 1.0, size=self.action_space.shape[0])
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size: #to ensure that you don't only learn from too little states at the beginning
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space.shape[0], dtype=np.int8)
        action_indices = np.dot(action, action_values) #dot product

        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)
        
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        q_target[batch_index, action_indices] = reward + self.gamma*np.max(q_next, axis=1)*done

        _ = self.q_eval.fit(state, q_target, verbose=0)

        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min


        def save_model(self):
            self.q_eval.save(self.model_file)
        
        def load_model(self):
            self.q_eval = load_model(self.model_file)


