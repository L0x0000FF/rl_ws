import numpy as np
import matplotlib.pyplot as plt

random_seed = 10

class multiarm_bandit:
  def __init__(self, k):
    r = np.random.RandomState(random_seed)
    self.k = k
    self.p = r.uniform(0,1,k)
    self.pmax = np.max(self.p)

  def action(self, index):
    result = np.random.random()
    if result < self.p[index]:
      result = 1
    else:
      result = 0
    return result
  
class solver:
  def __init__(self,bandit,epsilon = 0.01,init_prob = 1.0):
    self.bandit = bandit
    self.counts = np.zeros(bandit.k)
    self.regret = 0
    self.m = 0
    self.regrets = []
    self.actions = []
    self.reward = 0
    self.rewards = []
    self.epsilon = epsilon
    self.est = np.ones(bandit.k) * init_prob
  
  def update_regret(self,action):
    self.regret = self.regret + self.bandit.pmax - self.bandit.p[action]
    self.regrets.append(self.regret)
  
  def step(self):
    r = 0 
    if self.m >= self.bandit.k:
      tmp = np.random.random()
      if tmp < 1 - self.epsilon:
        a = np.argmax(self.est)
      else:
        a = np.random.randint(0,self.bandit.k)
      #self.epsilon = self.epsilon / 2
    else:
      a = self.m
      self.m = self.m + 1
    r = self.bandit.action(a)
    self.est[a] = (self.est[a] * self.counts[a] + r) / (self.counts[a] + 1)
    self.counts[a] = self.counts[a] + 1
    self.actions.append(a)
    self.update_regret(a)
    self.reward = self.reward + r
    self.rewards.append(self.reward)
  
mb = multiarm_bandit(10)
sl = solver(mb,epsilon=0.1)
for i in range(1,1000):
  sl.step()
print(mb.p)
print(sl.est)
print(sl.counts)
print(sl.epsilon)
print(sl.reward)
plt.plot(sl.regrets)
plt.show()
