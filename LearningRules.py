# Learning Rules #
import math


def computeNet(input, weights):
    net = 0
    for i in range(len(input)):
        net = net + input[i] * weights[i]
    print("NET:")
    print(net)
    return net


def computeFNetBinary(net):
    f_net = 0
    if (net > 0):
        f_net = 1
    if (net < 0):
        f_net = -1
    return f_net


def computeFNetCont(net):
    f_net = 0
    f_net = (2 / (1 + math.exp(-net))) - 1
    return f_net


def hebb(f_net):
    return f_net


def perceptron(desired, actual):
    return (desired - actual)

def correlation(desired,actual):
    return desired

def widrow(desired, actual):
    return (desired - actual)


def adjustWeights(inputs, weights, last, binary, desired, rule):
    c = 1
    if (last):
        print("COMPLETE")
        return
    current_input = inputs[0]
    inputs = inputs[1:]
    if desired:
        current_desired = desired[0]
        desired = desired[1:]
    if len(inputs) == 0:
        last = True
    net = computeNet(current_input, weights)
    if (binary):
        f_net = computeFNetBinary(net)
    else:
        f_net = computeFNetCont(net)
    if rule == "hebb":
        r = hebb(f_net)
    elif rule == "perceptron":
        r = perceptron(current_desired, f_net)
    elif rule == "widrow":
        r = widrow(current_desired, net)
    elif rule == "correlation":
        r = correlation(current_desired, net)

    del_weights = []
    for i in range(len(current_input)):
        x = (c * r) * current_input[i]
        del_weights.append(x)
        weights[i] = x + current_input[i]
#added the new code
    print("NEW WEIGHTS:")
    print(weights)
    adjustWeights(inputs, weights, last, binary, desired, rule)


if __name__ == "__main__":
    # total_inputs = (int)raw_input("Enter Total Number of Inputs)
    # vector_length = (int)raw_input("Enter Length of vector)
    total_inputs = 3
    vector_length = 4
    # for i in range(vector_length):
    # weight.append(raw_input("Enter Initial Weight:")
    weights = [1, -1, 0, 0.5]
    inputs = [[1, -2, 1.5, 0], [1, -0.5, -2, -1.5], [0, 1, -1, 1.5]]
    desired = [1, 2, 1, -1]
    print("BINARY HEBB!")
    adjustWeights(inputs, [1, -1, 0, 0.5], False, True, None, "hebb")
    print("CONTINUOUS HEBB!")
    adjustWeights(inputs, [1, -1, 0, 0.5], False, False, None, "hebb")
    print("PERCEPTRON!")
    adjustWeights(inputs, [1, -1, 0, 0.5], False, True, desired, "perceptron")
    print("WIDROW HOFF!")
    adjustWeights(inputs, [1, -1, 0, 0.5], False, True, desired, "widrow")
    print("CORRELATION!")
    adjustWeights(inputs, [1, -1, 0, 0.5], False, True, desired, "correlation")
