import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

# 策略网络和价值网络定义
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

#数据收集函数
def collect_data(env, policy, num_steps):
    state = env.reset()
    log_probs = []
    values = []
    states = []
    actions = []
    rewards = []
    masks = []
    for _ in range(num_steps):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, value = policy(state)
        action = torch.distributions.Categorical(action_probs).sample()
        next_state, reward, done, _ = env.step(action.item())

        log_prob = torch.log(action_probs.squeeze(0)[action])
        log_probs.append(log_prob)
        values.append(value.squeeze(1))
        rewards.append(torch.tensor([reward], dtype=torch.float))
        masks.append(torch.tensor([1-done], dtype=torch.float))
        states.append(state)
        actions.append(action)

        state = next_state
        if done:
            state = env.reset()

    return states, actions, log_probs, rewards, values, masks


# 优势函数和回报的计算
def compute_returns(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    returns = []
    advantages = []
    gae = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        next_value = values[step]
        returns.insert(0, gae + values[step])
        advantages.insert(0, gae)
    return returns, advantages


# 更新策略
def ppo_update(states, actions, log_probs, returns, advantages, policy, optimizer, ppo_epochs=4, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_prob, return_, advantage in zip(states, actions, log_probs, returns, advantages):
            action_prob, value = policy(state)
            dist = torch.distributions.Categorical(action_prob)
            new_log_prob = dist.log_prob(action)

            ratio = (new_log_prob - old_log_prob).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

