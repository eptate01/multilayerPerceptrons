import numpy as np
import math
import random

alpha = .5
class neuralNet():
    def __init__(self, hidden_nodes, input_amt):
        self.hidden_nodes_amt = hidden_nodes
        self.input_amt = input_amt
        self.input_weights = []
        self.hidden_weights = []
        self.in_bias = []
        self.hidden_bias = []
        

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

def rand_weights(network):
    
    for i in range(network.input_amt):
        network.input_weights.append(list(np.random.uniform(-1,1,network.hidden_nodes_amt)))
    for j in range(network.hidden_nodes_amt):
        network.hidden_weights.append(np.random.uniform(-1,1))
        network.in_bias.append(np.random.uniform(-1,1))
    network.hidden_bias.append(np.random.uniform(-1,1))
    

def forward_pass(inputs, network):
    network.h_in = np.add(np.dot(list(np.array(network.input_weights).transpose()), inputs), network.in_bias)
    network.h_out = list(map(sigmoid,network.h_in))
    raw_out = np.add(np.dot(network.hidden_weights, network.h_out),network.hidden_bias)
    output = sigmoid(raw_out)
    return output, network.h_in, network.h_out, raw_out

def back_pass(ans, inputs, network, output, h_in, h_out, raw_out):
    delta0 = (ans-output)*derivateSigmoid(raw_out[0])
    deltasLayer1 = np.multiply((delta0*np.array(network.hidden_weights)),list(map(derivateSigmoid,h_in)))
    network.hidden_weights = network.hidden_weights+alpha*np.multiply(h_out, delta0)
    network.input_weights = network.input_weights + alpha*np.multiply(np.matrix(inputs).T, np.matrix(deltasLayer1))
    network.in_bias = np.add(network.in_bias, alpha*deltasLayer1)
    network.hidden_bias[0] = np.add(network.hidden_bias[0], alpha*delta0)

def test(inputs, network):
    h_in = np.add(np.dot(list(np.array(network.input_weights).transpose()), inputs), network.in_bias)
    h_out = list(map(sigmoid,h_in))
    raw_out = np.add(np.dot(network.hidden_weights, h_out),network.hidden_bias)
    output = sigmoid(raw_out)
    return output

def run_update(ans, inputs, network):
    output, h_in, h_out, raw_out = forward_pass(inputs, network)
    back_pass(ans, inputs, network, output, h_in, h_out, raw_out)


#set beginning amounts
file = "mnist_train_0_1.csv"
inputList = openFile(file)
network = neuralNet(100, 784)
rand_weights(network)

#update weights and bias
for i in inputList:
    run_update(i[0],i[1:], network)

testFile = "mnist_test_0_1.csv"
TestList = openFile(testFile)
correct = 0
for i in TestList:
    expected = test(i[1:], network)
    if (expected > .5 and expected <1.5 and i[0] == 1) or (expected < .5 and i[0] == 0) or (expected > 1.5 and expected < 2.5 and i[0] == 2)or (expected > 2.5 and expected < 3.5 and i[0] == 3)or (expected > 3.5 and i[0] == 4):
        correct += 1
print(correct/len(TestList))
