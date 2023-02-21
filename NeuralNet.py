import numpy as np
import math
import random

def sigmoid(x):
    return 1/(1+math.pow(math.e,-x))

def rand_weights(inputs, hidden_nodes_amt, input_weights, hidden_weights, bias):
    
    for i in range(inputs):
        input_weights.append(list(np.random.uniform(-1,1,3)))
    for j in range(hidden_nodes_amt):
        hidden_weights.append(np.random.uniform(-1,1))
        bias.append(np.random.uniform(-1,1))

def forward_pass(inputs, input_weights, hidden_weights, bias):
    h_in = np.add(np.dot(list(np.array(input_weights).transpose()), inputs), bias)
    h_out = list(map(sigmoid,h_in))
    output = sigmoid(np.dot(hidden_weights, h_out))
    return output

def back_pass(ans, inputs, input_weights,hidden_weights, bias, output):
    error = ans - output
    

def run_update(ans, inputs, input_weights, hidden_weights, bias):
    output = forward_pass(inputs, input_weights,hidden_weights, bias)
    back_pass(ans, inputs, input_weights,hidden_weights, bias, output)


#set beginning amounts
hidden_nodes_amt = 3
input_amt = 4
input_weights = []
hidden_weights = []
bias = []
rand_weights(input_amt, hidden_nodes_amt, input_weights, hidden_weights, bias)

#grab one input
#4x1, will be 256x1
inputs = [22,24,23,4]
ans = 1

#update weights and bias
run_update(ans,[x for x in inputs], input_weights, hidden_weights, bias)