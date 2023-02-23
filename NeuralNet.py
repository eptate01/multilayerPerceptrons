import numpy as np
import math
import random

alpha = .5

def sigmoid(x):
    return 1/(1+math.pow(math.e,-x))

def derivateSigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def rand_weights(input_amt, hidden_nodes_amt, input_weights, hidden_weights, in_bias, hidden_bias):
    
    for i in range(input_amt):
        input_weights.append(list(np.random.uniform(-1,1,3)))
    for j in range(hidden_nodes_amt):
        hidden_weights.append(np.random.uniform(-1,1))
        in_bias.append(np.random.uniform(-1,1))
    hidden_bias.append(np.random.uniform(-1,1))
    

def forward_pass(inputs, input_weights, hidden_weights, in_bias,hidden_bias):
    h_in = np.add(np.dot(list(np.array(input_weights).transpose()), inputs), in_bias)
    h_out = list(map(sigmoid,h_in))
    output = sigmoid(np.add(np.dot(hidden_weights, h_out),hidden_bias))
    return output, h_in

def back_pass(ans, inputs, input_weights, hidden_weights, in_bias, hidden_bias, output, h_in):
    delta0 = (ans-output)*derivateSigmoid(output)
    deltasLayer1 = np.multiply((delta0*np.array(hidden_weights)), list(map(derivateSigmoid,h_in)))
    hidden_weights = hidden_weights+alpha*np.multiply(h_in, delta0)
    input_weights = input_weights + alpha*np.multiply(inputs, deltasLayer1)
    print(input_weights)



def run_update(ans, inputs, input_weights, hidden_weights, in_bias, hidden_bias):
    output, h_in = forward_pass(inputs, input_weights,hidden_weights, in_bias, hidden_bias)
    back_pass(ans, inputs, input_weights,hidden_weights, in_bias, hidden_bias, output, h_in)


#set beginning amounts
hidden_nodes_amt = 3
input_amt = 4
input_weights = []
hidden_weights = []
in_bias = []
hidden_bias = []
rand_weights(input_amt, hidden_nodes_amt, input_weights, hidden_weights, in_bias, hidden_bias)

#grab one input
#4x1, will be 256x1
inputs = [2,4,3,4]
ans = 1

#update weights and bias
run_update(ans,[x for x in inputs], input_weights, hidden_weights, in_bias, hidden_bias)