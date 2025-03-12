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
N = 20
hidden_dim = 128
total_episodes = 1000
max_steps = 10000
lr = 0.0005
lr_a = 0.001
lr_v = 0.0001
max_grad_norm = 0.5
gamma = 0.97
max_epoch = 10

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

  # def getAdvantages(self,rewards:torch.Tensor,values:torch.Tensor):
  #   with torch.no_grad():
  #       advantages = torch.zeros_like(rewards)
  #       rt = torch.zeros_like(values)
  #       # 从后往前累计奖励
  #       for t in reversed(range(len(rewards))):
  #           if t + 1 < len(rewards):
  #               rt[t] = rewards[t] + gamma * rt[t+1]
  #           else:
  #               rt[t] = rewards[t]
  #       advantages = rt - values
  #   return advantages, rt

  def update(self):
    for epoch in range(max_epoch):
      la = []
      lv = []
      states,next_states,actions,log_probs,values,rewards,dones,batches = self.mem.replay()
      states = torch.tensor(states,dtype=torch.float).to(device)
      next_states = torch.tensor(next_states,dtype=torch.float).to(device)
      rewards = torch.tensor(rewards,dtype=torch.float).to(device)
      actions = torch.tensor(actions,dtype=torch.int64).to(device)
      log_probs = torch.tensor(log_probs,dtype=torch.float).to(device)
      values = torch.tensor(values,dtype=torch.float).to(device)
      
      # print(values.shape,self.critic(next_states).squeeze(1).shape)
      with torch.no_grad():
        target_v = rewards + gamma * self.critic(next_states).squeeze(1)
      advantages = (target_v - self.critic(states).squeeze(1)).detach()
      # print(advantages.shape)
      for batch in batches:
        # batch
        s_batch = states[batch]
        a_batch = actions[batch]
        v_batch = values[batch]
        target_v_batch = target_v[batch]
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

        loss_a = -torch.min(surr1,surr2)
        self.actor.opt.zero_grad()
        loss_a.mean().backward()
        la.append(loss_a.mean().item())
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        self.actor.opt.step()
        # update critic
        new_v_batch = self.critic(s_batch).squeeze(1)
        loss_v = F.mse_loss(new_v_batch,target_v_batch)
        lv.append(loss_v.mean().item())
        self.critic.opt.zero_grad()
        loss_v.mean().backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.critic.opt.step()
      self.loss_a_mean.append(np.sum(la))
      self.loss_v_mean.append(np.sum(lv))
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


