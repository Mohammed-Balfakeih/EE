from test import Agent
import numpy as np
from utils import plotLearning
import gym
from gym import spaces

if __name__ == '__main__':
    env = gym.make('BipedalWalkerHardcore-v3')
    n_games = 500
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=24,
                 mem_size=1000000, batch_size=64, action_space=spaces.Box(low=-1.0, high=1.0, shape = (4,), dtype=np.float32), epsilon_end=0.01)
    #agent.load_model()
    scores = []
    eps_history = []
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            print("\n")
            print(action)
            print("\n")
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('epsiode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score)
        if i % 10 == 0 and i > 0:
            agent.save_model()
        
        filename='BipedalWalker.png'
        x = [i+1 for i in range(n_games)]
        print(x)
        plotLearning(x, scores, eps_history, filename)
        