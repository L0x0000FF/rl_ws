import gym 
import numpy as np
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

env = gym.make("CartPole-v1",render_mode='rgb_array')

env.reset()

a_size = env.action_space.n
s_size = env.observation_space.shape[0]
N = 20
hidden_dim = 64
total_episodes = 5000
max_steps = 10000
lr_a = 0.0001
lr_v = 0.001
gamma = 0.99
max_epoch = 5

epsilon = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOMemory:
  def __init__(self,batch_size):
    self.states = []
    self.log_probs = []
    self.actions = []
    self.rewards = []
    self.dones = []
    self.values = []
    self.batch_size = batch_size
  
  def remember(self,state,reward,action,log_prob,value,done):
    self.states.append(state)
    self.rewards.append(reward)
    self.actions.append(action)
    self.log_probs.append(log_prob)
    self.values.append(value)
    self.dones.append(done)
  
  def replay(self):
    n_states = len(self.states)
    batch_start = np.arange(0, n_states, self.batch_size)
    indices = np.arange(n_states, dtype=np.int64)
    np.random.shuffle(indices)
    batches = [indices[i:i + self.batch_size] for i in batch_start]

    return np.array(self.states), \
           np.array(self.actions), \
           np.array(self.log_probs), \
           np.array(self.values), \
           np.array(self.rewards), \
           np.array(self.dones), \
           batches

  def clearMem(self):
    self.states.clear()
    self.actions.clear()
    self.values.clear()
    self.rewards.clear()
    self.dones.clear()
    self.log_probs.clear()

class PolicyNet(nn.Module):
  def __init__(self,hidden_dim=32):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(s_size,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,a_size))
    self.opt = torch.optim.AdamW(self.parameters(),lr=lr_a)
  
  def forward(self,input):
    if type(input) != torch.Tensor:
      input = torch.tensor(input)
    return self.net(input)
  
class CriticNet(nn.Module):
  def __init__(self,hidden_dim=32):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(s_size,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,1))
    self.opt = torch.optim.AdamW(self.parameters(),lr=lr_v)
  
  def forward(self,input):
    if type(input) != torch.Tensor:
      input = torch.tensor(input)
    return self.net(input)
  
class PPOAgent():
  def __init__(self):
    self.actor = PolicyNet(hidden_dim=hidden_dim).to(device)
    self.critic = CriticNet(hidden_dim=hidden_dim).to(device)
    self.mem = PPOMemory(batch_size=64)

  def getAction(self,state):
    with torch.no_grad():
      if type(state) != torch.Tensor:
        state = torch.tensor(state).to(device)
      state = state.unsqueeze(dim=0)
      # print(states)
      logits = self.actor(state)
      # print(logits)
      probs = F.softmax(logits,dim=-1)
      # print(probs)
      action = torch.multinomial(probs,num_samples=1)
      action = action.squeeze(0)
      # print(action,logits)
      log_prob = torch.log(probs.gather(dim=1,index=action.unsqueeze(1))).squeeze(-1)
      return action.tolist()[0],log_prob

  def getAdvantages(self,rewards:torch.Tensor,values:torch.Tensor):
    with torch.no_grad():
        advantages = torch.zeros_like(rewards)
        rt = torch.zeros_like(values)
        # 从后往前累计奖励
        for t in reversed(range(len(rewards))):
            if t + 1 < len(rewards):
                rt[t] = rewards[t] + gamma * rt[t+1]
            else:
                rt[t] = rewards[t]
        advantages = rt - values
    return advantages, rt

  def update(self):
    for epoch in range(max_epoch):
      states,actions,log_probs,values,rewards,dones,batches = self.mem.replay()

      rewards = torch.tensor(rewards,dtype=torch.float).to(device)
      values = torch.tensor(values,dtype=torch.float).to(device)
      adv,rt = self.getAdvantages(rewards,values)
      adv = (adv - adv.mean()) / (adv.std() + 1e-8)

      for batch in batches:
        states_batch = torch.tensor(states[batch],dtype=torch.float).to(device)
        actions_batch = torch.tensor(actions[batch],dtype=torch.int64).to(device)
        log_probs_batch = torch.tensor(log_probs[batch],dtype=torch.float).to(device)

        logits = self.actor(states_batch)
        new_probs = F.softmax(logits,dim=-1)
        # print(logits,new_probs)
        new_log_probs = torch.log(new_probs)
        new_log_probs = new_log_probs.gather(dim=1,index=actions_batch.unsqueeze(1))
        new_log_probs = new_log_probs.squeeze(-1)
        # print(new_log_probs.shape,log_probs_batch.shape)
        ratio = torch.exp(new_log_probs - log_probs_batch)
        surr1 = ratio * adv[batch].squeeze(1)
        surr2 = torch.clamp(ratio,min=1-epsilon,max=1+epsilon) * adv[batch].squeeze(1)

        loss_a = -torch.min(surr1,surr2)
        new_values = self.critic(states_batch).squeeze(-1)
        loss_v = F.mse_loss(new_values,rt[batch].squeeze(1))
        # print(new_values.shape,rt[batch].shape,loss_v.shape)
        
        
        self.actor.opt.zero_grad()
        self.critic.opt.zero_grad()
        loss_a.mean().backward()
        loss_v.backward()
        self.actor.opt.step()
        self.critic.opt.step()
    self.mem.clearMem()

agent = PPOAgent()

score_all = []
for episode in tqdm(range(total_episodes)):
  s,_ = env.reset()
  done = False
  steps = 0
  score = 0
  while not done:
    steps += 1
    # print(s)
    action,log_prob = agent.getAction(s)
    # print(action,log_prob)
    # print(action,torch.exp(log_prob))
    new_s,reward,done,truncated,info = env.step(action)
    value = agent.critic(s).detach()
    agent.mem.remember(s,reward,action,log_prob,value,done)
    s = new_s
    score += reward
    if steps % N == 0:
      agent.update()
  score_all.append(score)
  tqdm.write("episode %i,score=%f"%(episode,score))
      
plt.plot(score_all)
plt.show()


