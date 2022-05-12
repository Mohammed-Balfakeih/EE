from random import sample
import gym
import numpy as np

env = gym.make('BipedalWalkerHardcore-v3')
env.reset()

for _ in range(1000):
    obs, reward, done, info = env.step((env.action_space.sample())) # take a random action
    print(reward)
    if(done):
        break
    env.render()


