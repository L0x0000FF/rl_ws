import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

a = nn.Sequential(nn.Linear(1,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,1))

def f(x):
  return x + np.exp(0.1 * x) * 0.1 + np.sqrt(x)

x = np.arange(0,100,0.1,dtype=float)
y = f(x)# + np.random.random(size=x.shape)

x = torch.tensor(x,dtype=torch.float)
y = torch.tensor(y,dtype=torch.float)

def gen_batch(x,y,batch_size=32):
  l = len(x)
  batch_start = np.arange(0, l,batch_size)
  indices = np.arange(l, dtype=np.int64)
  np.random.shuffle(indices)
  batches = [indices[i:i + batch_size] for i in batch_start]
  return batches

# print(x.unsqueeze(1).shape)
opt = torch.optim.AdamW(a.parameters(),lr=0.02)
for epoch in range(500):
  batches = gen_batch(x,y)
  for batch in batches:
    x_batch = x[batch]
    y_batch = y[batch]
    out = a(x_batch.unsqueeze(1))
    opt.zero_grad()
    loss = nn.functional.mse_loss(out.squeeze(1),y_batch)
    loss.backward()
    opt.step()

plt.plot(a(x.unsqueeze(1)).tolist(),'-')
plt.plot(y.tolist(),'.')
plt.show()