import math
import sys
from src.matrix import *

class layer:
    def __init__(self):
        self.inp = None
        self.w = None
        self.dw = None
        self.v = None
        self.out = None
        self.activation = ""

class model:
    def __init__(self):
        self.n = 0
        self.layers = []

# Run an activation function on each element in a matrix,
# modifies the matrix in place
# matrix m: Input to activation function
# ACTIVATION a: function to run
def activate_matrix(m, a):
    for i in range(m.rows):
        sum_val = 0.0
        for j in range(m.cols):
            x = m.data[i][j]
            if a == 'LOGISTIC':
                m.data[i][j] = 1 / (1 + math.exp(-x))
            elif a == 'RELU':
                m.data[i][j] = max(0, x)
            elif a == 'LRELU':
                m.data[i][j] = max(0.1 * x, x)
            elif a == 'SOFTMAX':
                m.data[i][j] = math.exp(x)
            sum_val += m.data[i][j]

        if a == 'SOFTMAX':
            for j in range(m.cols):
                m.data[i][j] /= sum_val

# Calculates the gradient of an activation function and multiplies it into
# the delta for a layer
# matrix m: an activated layer output
# ACTIVATION a: activation function for a layer
# matrix d: delta before activation gradient
def gradient_matrix(m, a, d):
    for i in range(m.rows):
        for j in range(m.cols):
            x = m.data[i][j]
            # TODO: multiply the correct element of d by the gradient
            if a == 'LOGISTIC':
                d.data[i][j] *= x * (1 - x)
            elif a == 'RELU':
                d.data[i][j] *= 1 if x > 0 else 0
            elif a == 'LRELU':
                d.data[i][j] *= 1 if x > 0 else 0.1

# Forward propagate information through a layer
# layer *l: pointer to the layer
# matrix in: input to layer
# returns: matrix that is output of the layer
def forward_layer(l, inp):
    l.inp = inp  # Save the input for backpropagation

    # TODO: fix this! multiply input by weights and apply activation function.
    out = make_matrix(inp.rows, l.w.cols)
    out = matrix_mult_matrix(inp, l.w)
    activate_matrix(out, l.activation)
    
    free_matrix(l.out)  # free the old output
    l.out = out         # Save the current output for gradient calculation
    return out

# Backward propagate derivatives through a layer
# layer *l: pointer to the layer
# matrix delta: partial derivative of loss w.r.t. output of layer
# returns: matrix, partial derivative of loss w.r.t. input to layer
def backward_layer(l, delta):
    # 1.4.1
    # delta is dL/dy
    # TODO: modify it in place to be dL/d(xw)
    gradient_matrix(l.out, l.activation, delta)

    # 1.4.2
    # TODO: then calculate dL/dw and save it in l->dw
    free_matrix(l.dw)
    dw = matrix_mult_matrix(transpose_matrix(l.inp), delta)
    l.dw = dw


    # 1.4.3
    # TODO: finally, calculate dL/dx and return it.
    dx = matrix_mult_matrix(delta, transpose_matrix(l.w))

    return dx

# Update the weights at layer l
# layer *l: pointer to the layer
# double rate: learning rate
# double momentum: amount of momentum to use
# double decay: value for weight decay
def update_layer(l, rate, momentum, decay):
    # TODO:
    # Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    # save it to l->v
    delta_w = axpy_matrix(-decay, l.w, l.dw)
    delta_w = axpy_matrix(momentum, l.v, delta_w)
    l.v = delta_w

    # Update l->w
    l.w = axpy_matrix(rate, l.v, l.w)

    # Remember to free any intermediate results to avoid memory leaks
    

# Make a new layer for our model
# int input: number of inputs to the layer
# int output: number of outputs from the layer
# ACTIVATION activation: the activation function to use
def make_layer(input, output, activation):
    l = layer()
    l.inp  = make_matrix(1, 1)
    l.out = make_matrix(1, 1)
    l.w   = random_matrix(input, output, math.sqrt(2.0 / input))
    l.v   = make_matrix(input, output)
    l.dw  = make_matrix(input, output)
    l.activation = activation
    return l

# Run a model on input X
# model m: model to run
# matrix X: input to model
# returns: result matrix
def forward_model(m, X):
    for i in range(m.n):
        X = forward_layer(m.layers[i], X)
    return X

# Run a model backward given gradient dL
# model m: model to run
# matrix dL: partial derivative of loss w.r.t. model output dL/dy
def backward_model(m, dL):
    d = copy_matrix(dL)
    for i in range(m.n - 1, -1, -1):
        prev = backward_layer(m.layers[i], d)
        free_matrix(d)
        d = prev
    free_matrix(d)

# Update the model weights
# model m: model to update
# double rate: learning rate
# double momentum: amount of momentum to use
# double decay: value for weight decay
def update_model(m, rate, momentum, decay):
    for i in range(m.n):
        update_layer(m.layers[i], rate, momentum, decay)

# Find the index of the maximum element in an array
# double *a: array
# int n: size of a, |a|
# returns: index of maximum element
def max_index(a, n):
    if n <= 0: return -1
    max_i = 0
    max_val = a[0]
    for i in range(1, n):
        if a[i] > max_val:
            max_val = a[i]
            max_i = i
    return max_i

# Calculate the accuracy of a model on some data d
# model m: model to run
# data d: data to run on
# returns: accuracy, number correct / total
def accuracy_model(m, d):
    p = forward_model(m, d.X)
    correct = 0
    for i in range(d.y.rows):
        if max_index(d.y.data[i], d.y.cols) == max_index(p.data[i], p.cols):
            correct += 1
    return float(correct) / d.y.rows

# Calculate the cross-entropy loss for a set of predictions
# matrix y: the correct values
# matrix p: the predictions
# returns: average cross-entropy loss over data points, 1/n Σ(-ylog(p))
def cross_entropy_loss(y, p):
    sum_val = 0.0
    for i in range(y.rows):
        for j in range(y.cols):
            sum_val += -y.data[i][j] * math.log(p.data[i][j])
    return sum_val / y.rows

# Train a model on a dataset using SGD
# model m: model to train
# data d: dataset to train on
# int batch: batch size for SGD
# int iters: number of iterations of SGD to run (i.e. how many batches)
# double rate: learning rate
# double momentum: momentum
# double decay: weight decay
def train_model(m, d, batch, iters, rate, momentum, decay):
    for e in range(iters):
        b = random_batch(d, batch)
        p = forward_model(m, b.X)
        print(f"{e:06d}: Loss: {cross_entropy_loss(b.y, p):.6f}", file=sys.stderr)
        dL = axpy_matrix(-1, p, b.y) # partial derivative of loss dL/dy
        backward_model(m, dL)
        update_model(m, rate / batch, momentum, decay)
        free_matrix(dL)
        free_data(b)


# Questions
#
# 2.1.1 What are the training and test accuracy values you get? Why might we be interested in both training accuracy and testing accuracy? What do these two numbers tell us about our current model?
# TODO
#
# 2.1.2 Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model. What patterns do you see and how does the choice of learning rate affect both the loss during training and the final model accuracy?
# TODO
#
# 2.1.3 Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay affect the final model training and test accuracy?
# TODO
#
# 2.2.1 Currently the model uses a logistic activation for the first layer. Try using a the different activation functions we programmed. How well do they perform? What's best?
# TODO
#
# 2.2.2 Using the same activation, find the best (power of 10) learning rate for your model. What is the training accuracy and testing accuracy?
# TODO
#
# 2.2.3 Right now the regularization parameter `decay` is set to 0. Try adding some decay to your model. What happens, does it help? Why or why not may this be?
# TODO
#
# 2.2.4 Modify your model so it has 3 layers instead of two. The layers should be `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify your model to train for 3000 iterations instead of 1000. Look at the training and testing error for different values of decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?
# TODO
#
# 3.1.1 What is the best training accuracy and testing accuracy? Summarize all the hyperparameter combinations you tried.
# TODO
#