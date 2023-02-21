import numpy as np
import math
import random
def sigmoid(x):
    return 1/(1+math.pow(math.e,-x))
#The architecture is 4 - 3 - 1
#4x1
input = [1,2,3,4]
output = [-1]
#4x3
weights_h = []
#3x1
weights_o = []
for i in range(len(input)):
    weights_h.append(list(np.random.uniform(-1,1,3)))
for j in range(3):
    weights_o.append(random.uniform(-1,1))
print(weights_h)
print(weights_o)