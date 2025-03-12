import gym 
import numpy as np
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make("CartPole-v1",render_mode='rgb_array')

env.reset()

a_size = env.action_space.n
s_size = env.observation_space

total_episodes = 1000
max_steps = 10000
learning_rate = 0.001
gamma = 0.99

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyPi(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.hidden = nn.Linear(4, hidden_dim)
        self.classify = nn.Linear(hidden_dim, 2)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        logits = self.classify(outs)
        return logits
    
policy = PolicyPi().to(device)
opt = torch.optim.AdamW(policy.parameters(),lr = learning_rate)

def get_action(s):
  with torch.no_grad():
    s_batch = np.expand_dims(s,axis=0)
    s_batch = torch.tensor(s_batch,dtype=torch.float).to(device)
    logit = policy(s_batch)
    logit = logit.squeeze(dim=0)
    prob = F.softmax(logit,dim=-1)
    a = torch.multinomial(prob,num_samples=1)
    return a.tolist()[0]

rewards_all = []
for episode in range(total_episodes):
  cumlative_reward = []
  actions = []
  states = []
  state,_ = env.reset()
  rewards = []
  while True:
    states.append(state)
    a = get_action(state)
    new_state,reward,done,truncated,info = env.step(a)    
    rewards.append(reward)
    actions.append(a)
    state = new_state
    if done or truncated:
      break
  rewards_all.append(np.sum(rewards))
  cumlative_reward = np.zeros_like(rewards)
  for i in reversed(range(len(rewards))):
    cumlative_reward[i] = rewards[i] + (cumlative_reward[i+1]*gamma if i+1 < len(rewards) else 0)
  
  states = torch.tensor(states, dtype=torch.float).to(device)
  actions = torch.tensor(actions, dtype=torch.int64).to(device)
  cumlative_reward = torch.tensor(cumlative_reward, dtype=torch.float).to(device)

  opt.zero_grad()
  logit = policy(states)
  log_probs = F.cross_entropy(logit,actions,reduction="none")
  loss = log_probs * cumlative_reward
  loss.sum().backward()
  opt.step()
  
  print("episode",episode,"reward=",rewards_all[-1])

env.close()  

plt.plot(rewards_all)
plt.show()

