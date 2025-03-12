import gym
import numpy as np
import random

env = gym.make("FrozenLake-v1",render_mode='human')

a_size = env.action_space.n
s_size = env.observation_space.n
q_table = np.zeros((s_size, a_size))

total_episodes = 15000
max_steps = 99
learning_rate = 0.9
gamma = 0.95

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.01

rewards = []
for episode in range(total_episodes):
  state = env.reset()
  state = state[0]
  step = 0
  done = False
  total_rewards = 0
  for step in range(max_steps):
    if random.uniform(0,1) < epsilon:
      #explore
      action = env.action_space.sample()
    else:
      #exploit
      action = np.argmax(q_table[state])
    new_state, reward, done,_, info = env.step(action)
    q_table[state][action] = (1 - learning_rate) * q_table[state][action] + learning_rate * (reward + gamma * np.max(q_table[new_state]))
    total_rewards += reward
    state = new_state
    if done == True:
      break
    epsilon -= epsilon_decay
    if epsilon < min_epsilon:
      epsilon
  rewards.append(total_rewards)

print(rewards)
print(q_table)




