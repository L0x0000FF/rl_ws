import numpy as np

n_states = 129
batch_size = 64
batch_start = np.arange(0, n_states, batch_size)
indices = np.arange(n_states, dtype=np.int64)
np.random.shuffle(indices)
batches = [indices[i:i + batch_size] for i in batch_start]
print(indices)
print(batches)
a = 1
b = 2.3
print("a%i%f"%(a,b))