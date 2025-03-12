import numpy as np
import torch
import torch.nn as nn

a = nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,1))
b = torch.tensor([[1,2,3,4],[5,6,7,8],[9,0,1,2]],dtype=torch.float)
c = b.unsqueeze(0)
print(b,"\n",c)
print(a(b),"\n",a(c))
d = torch.tensor([1,2,3,4,5])
print(d.unsqueeze(0))
print(d.unsqueeze(1))