import numpy as np

minibatch = [ 0.94, 0.92, 0.92, -0.26, 0.24, 0.24]
minibatch = np.array(minibatch)

action_m = np.array([ 2.,  1. , 0.,  3.,  3.,  1.,  1.,  2.,  3. , 0.  ,0.  ,1. , 3. , 0. , 0. , 0. , 3.])

# action_m = action_m.astype(int)
y_train = np.zeros((100, 1))

y_train[action_m, 0] = 3
