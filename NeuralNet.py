import numpy as np
import math
import random

alpha = .5
class neuralNet:
    
def openFile(fileName): #Opens the file and reads in the data
    with open(fileName, "r") as file:
        data = file.readlines()
        size = len(data)
        for i in range(0,size):
            data[i] = data[i].rstrip("\n")
            data[i] = data[i].split(",")
            data[i][0] = int(data[i][0])
            for x in range(1,len(data[i])):
                data[i][x] = float(data[i][x])/100
    return data

def sigmoid(x):
    return 1/(1+math.pow(math.e,-x))

def derivateSigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def rand_weights(input_amt, hidden_nodes_amt, input_weights, hidden_weights, in_bias, hidden_bias):
    
    for i in range(input_amt):
        input_weights.append(list(np.random.uniform(-1,1,hidden_nodes_amt)))
    for j in range(hidden_nodes_amt):
        hidden_weights.append(np.random.uniform(-1,1))
        in_bias.append(np.random.uniform(-1,1))
    hidden_bias.append(np.random.uniform(-1,1))
    

def forward_pass(inputs, input_weights, hidden_weights, in_bias,hidden_bias):
    h_in = np.add(np.dot(list(np.array(input_weights).transpose()), inputs), in_bias)
    h_out = list(map(sigmoid,h_in))
    raw_out = np.add(np.dot(hidden_weights, h_out),hidden_bias)
    output = sigmoid(raw_out)
    return output, h_in, h_out, raw_out

def back_pass(ans, inputs, input_weights, hidden_weights, in_bias, hidden_bias, output, h_in, h_out, raw_out):
    delta0 = (ans-output)*derivateSigmoid(raw_out[0])
    deltasLayer1 = np.multiply(list(map(derivateSigmoid,h_in)),(delta0*np.array(hidden_weights)))
    hidden_weights = hidden_weights+alpha*np.multiply(h_out, delta0)
    input_weights = input_weights + alpha*np.multiply(np.matrix(inputs).T, np.matrix(deltasLayer1))
    in_bias = np.add(in_bias, alpha*deltasLayer1)
    hidden_bias[0] = np.add(hidden_bias[0], alpha*delta0)

def test(inputs, input_weights, hidden_weights, in_bias, hidden_bias):
    h_in = np.add(np.dot(list(np.array(input_weights).transpose()), inputs), in_bias)
    h_out = list(map(sigmoid,h_in))
    raw_out = np.add(np.dot(hidden_weights, h_out),hidden_bias)
    output = sigmoid(raw_out)
    return output

def run_update(ans, inputs, input_weights, hidden_weights, in_bias, hidden_bias):
    output, h_in, h_out, raw_out = forward_pass(inputs, input_weights,hidden_weights, in_bias, hidden_bias)
    back_pass(ans, inputs, input_weights,hidden_weights, in_bias, hidden_bias, output, h_in, h_out, raw_out)


#set beginning amounts
file = "mnist_train_0_1.csv"
inputList = openFile(file)
hidden_nodes_amt = 5
input_amt = 784
input_weights = []
hidden_weights = []
in_bias = []
hidden_bias = []
rand_weights(input_amt, hidden_nodes_amt, input_weights, hidden_weights, in_bias, hidden_bias)

#grab one input
#4x1, will be 256x1

#update weights and bias
for i in inputList:
    run_update(i[0],i[1:], input_weights, hidden_weights, in_bias, hidden_bias)

testFile = "mnist_test_0_1.csv"
TestList = openFile(testFile)
correct = 0
for i in TestList:
    expected = test(i[1:], input_weights, hidden_weights, in_bias, hidden_bias)
    if (expected > .5 and i[0] == 1) or (expected < .5 and i[0] == 0):
        correct += 1
print(correct/len(TestList))
