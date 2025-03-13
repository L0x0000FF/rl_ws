import gymnasium as gym
import numpy as np
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

env = gym.make("CartPole-v1",render_mode="rgb_array")

env.reset()

a_size = env.action_space.n
s_size = env.observation_space.shape[0]
max_mem_size = 128
N = 20
hidden_dim = 128
total_episodes = 2000
max_steps = 10000
lr = 0.001
lr_a = 0.0003
lr_v = 0.001
max_grad_norm = 0.5
gamma = 0.97
max_epoch = 3

epsilon = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOMemory:
  def __init__(self,batch_size):
    # nx1 lists
    self.states = []
    self.next_states = []
    self.log_probs = []
    self.actions = []
    self.rewards = []
    self.dones = []
    self.values = []
    self.batch_size = batch_size
  
  def remember(self,state,next_state,reward,action,log_prob,value,done):
    self.states.append(state)
    self.next_states.append(next_state)
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
           np.array(self.next_states), \
           np.array(self.actions), \
           np.array(self.log_probs), \
           np.array(self.values), \
           np.array(self.rewards), \
           np.array(self.dones), \
           batches

  def clearMem(self):
    self.states.clear()
    self.next_states.clear()
    self.actions.clear()
    self.values.clear()
    self.rewards.clear()
    self.dones.clear()
    self.log_probs.clear()

  def size(self):
    return len(self.states)

class PolicyNet(nn.Module):
  def __init__(self,hidden_dim=32):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(s_size,hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim,hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim,a_size))
    self.opt = torch.optim.AdamW(self.parameters(),lr=lr_a)
  
  def forward(self,input):
    if type(input) != torch.Tensor:
      input = torch.tensor(input)
    return self.net(input)
  
class CriticNet(nn.Module):
  def __init__(self,hidden_dim=32):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(s_size,hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim,hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim,1))
    self.opt = torch.optim.AdamW(self.parameters(),lr=lr_v)
  
  def forward(self,input):
    if type(input) != torch.Tensor:
      input = torch.tensor(input).to(device)
    return self.net(input)
  
class PPOAgent():
  def __init__(self):
    self.actor = PolicyNet(hidden_dim=hidden_dim).to(device)
    self.critic = CriticNet(hidden_dim=hidden_dim).to(device)
    self.loss_a_mean = []
    self.loss_v_mean = []
    self.mem = PPOMemory(batch_size=64)

  def getAction(self,state):
    with torch.no_grad():
      if type(state) != torch.Tensor:
        state = torch.tensor(state).to(device)
      state = state.unsqueeze(dim=0)
      # print(states)
      logits = self.actor(state)
      # print(logits)
      probs = F.softmax(logits,dim=1)
      # print(probs)
      action = torch.multinomial(probs,num_samples=1)
      action = action.squeeze(0)
      # print(action,logits)
      log_prob = torch.log(probs)
      return action.tolist()[0],log_prob.tolist()[0]

  def getAdvantages_n_returns(self,rewards:torch.Tensor,values:torch.Tensor,dones:np.ndarray,gae_lambda = 0.9):
    with torch.no_grad():
        l = len(values)
        advantages = []
        rtgs = []
        next_adv = 0
        next_rtgs = 0
        for i in reversed(range(l)):
          if dones[i] == True or i+1 >= l:
            next_adv = 0
            next_rtgs = 0
            gae = rewards[i] - values[i]
          else:
            gae = rewards[i] + gamma * values[i+1] - values[i]
          advantages.insert(0,gae + gamma * gae_lambda * next_adv)
          rtgs.insert(0,rewards[i] + gamma * next_rtgs)
          next_adv = advantages[0]
          next_rtgs = rtgs[0]
    return torch.tensor(advantages,dtype=torch.float).to(device), torch.tensor(rtgs,dtype=torch.float).to(device)

  def update(self):
    states,next_states,actions,log_probs,values,rewards,dones,batches = self.mem.replay()
    for epoch in range(max_epoch):
      la = []
      lv = []
      states = torch.tensor(states,dtype=torch.float).to(device)
      next_states = torch.tensor(next_states,dtype=torch.float).to(device)
      rewards = torch.tensor(rewards,dtype=torch.float).to(device)
      actions = torch.tensor(actions,dtype=torch.int64).to(device)
      log_probs = torch.tensor(log_probs,dtype=torch.float).to(device)
      values = torch.tensor(values,dtype=torch.float).to(device)
      
      # with torch.no_grad():
      #   r_len = len(rewards)
      #   target_v = torch.zeros_like(values).to(device)
      #   for i in reversed(range(r_len)):
      #     target_v[i] = rewards[i] + (gamma * target_v[-1] if i+1 < r_len else 0)
      advantages,target_v = self.getAdvantages_n_returns(rewards,values,dones)#(target_v - values).detach()
      # print(advantages.shape)
      for batch in batches:
        # batch
        s_batch = states[batch]
        a_batch = actions[batch]
        v_batch = values[batch]
        target_v_batch = target_v[batch].detach()
        adv = advantages[batch]
        log_probs_batch = log_probs[batch]

        new_logits_batch = self.actor(s_batch)
        new_probs_batch = F.softmax(new_logits_batch,dim=1)

        # log_probs_batch:nx2 ,log_probs_batch:nx1
        log_probs_batch_a = log_probs_batch.gather(dim=1,index=a_batch.unsqueeze(1))
        # transform from nx1 to n
        log_probs_batch_a = log_probs_batch_a.squeeze(1)

        # same to log_probs_batch_a
        new_probs_batch_a = new_probs_batch.gather(dim=1,index=a_batch.unsqueeze(1))
        # to log prob
        new_log_probs_batch_a = torch.log(new_probs_batch_a)
        new_log_probs_batch_a = new_log_probs_batch_a.squeeze(1)

        ratio = torch.exp(new_log_probs_batch_a - log_probs_batch_a)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio,1-epsilon,1+epsilon) * adv

        # update critic
        new_v_batch = self.critic(s_batch).squeeze(1)
        loss_v = F.mse_loss(new_v_batch,target_v_batch)
        lv.append(loss_v.item())
        self.critic.opt.zero_grad()
        loss_v.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.critic.opt.step()
        
        loss_a = -torch.min(surr1,surr2)
        self.actor.opt.zero_grad()
        loss_a.mean().backward()
        la.append(loss_a.mean().item())
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        self.actor.opt.step()
        
      self.loss_a_mean.append(np.sum(la))
      self.loss_v_mean.append(np.sum(lv))
    if self.mem.size() >= max_mem_size:
      self.mem.clearMem()


agent = PPOAgent()
score_all = []
for episode in tqdm(range(total_episodes)):
  s,_ = env.reset()
  done = False
  truncated = False
  steps = 0
  score = 0
  while (not done) and (not truncated):
    steps += 1
    action,log_prob = agent.getAction(s)
    # print(action,log_prob)
    # print(action,torch.exp(log_prob))
    new_s,reward,done,truncated,info = env.step(action)
    value = agent.critic(s).detach()
    agent.mem.remember(s,new_s,reward,action,log_prob,value.item(),done)
    s = new_s
    score += reward
    if (steps + 1) % N == 0:
      agent.update()
  score_all.append(score)
  tqdm.write("episode %i,score=%f"%(episode,score))
      
plt.subplot(1,3,1)
plt.plot(score_all)      
plt.subplot(1,3,2)
plt.plot(agent.loss_a_mean)      
plt.subplot(1,3,3)
plt.plot(agent.loss_v_mean)
plt.show()


