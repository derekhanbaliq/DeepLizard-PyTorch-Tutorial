import numpy as np

t1 = np.array([1, 1, 1])
t2 = np.array([2, 2, 2])
t3 = np.array([3, 3, 3])

print(np.concatenate((t1, t2, t3), axis=0))
print(np.stack((t1, t2, t3), axis=0))

print(np.concatenate((np.expand_dims(t1, 0), np.expand_dims(t2, 0), np.expand_dims(t3, 0)), axis=0))
