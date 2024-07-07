import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from exercise_code.networks import ClassificationNet
from exercise_code.networks import Sigmoid, Relu

from exercise_code.networks.layer import affine_forward, affine_backward, Sigmoid, Tanh, LeakyRelu, Relu
from exercise_code.networks.base_networks import Network


class ClassificationNet(Network):
    """
    A fully-connected classification neural network with configurable 
    activation function, number of layers, number of classes, hidden size and
    regularization strength. 
    """

    def __init__(self, activation=Sigmoid, num_layer=2,
                 input_size=3 * 32 * 32, hidden_size=100,
                 std=1e-3, num_classes=10, reg=0, **kwargs):
        """
        :param activation: choice of activation function. It should implement
            a forward() and a backward() method.
        :param num_layer: integer, number of layers. 
        :param input_size: integer, the dimension D of the input data.
        :param hidden_size: integer, the number of neurons H in the hidden layer.
        :param std: float, standard deviation used for weight initialization.
        :param num_classes: integer, number of classes.
        :param reg: float, regularization strength.
        """
        super().__init__("cifar10_classification_net")

        self.activation = activation()
        self.reg_strength = reg

        self.cache = None

        self.memory = 0
        self.memory_forward = 0
        self.memory_backward = 0
        self.num_operation = 0

        # Initialize random gaussian weights for all layers and zero bias
        self.num_layer = num_layer
        self.std = std
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.reset_weights()

    def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: Input data of shape N x D. Each X[i] is a training sample.
        :return: Predicted value for the data in X, shape N x 1
                 1-dimensional array of length N with the classification scores.
        """

        self.cache = {}
        self.reg = {}
        X = X.reshape(X.shape[0], -1)
        # Unpack variables from the params dictionary
        for i in range(self.num_layer - 1):
            W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]

            # Forward i_th layer
            X, cache_affine = affine_forward(X, W, b)
            self.cache["affine" + str(i + 1)] = cache_affine

            # Activation function
            X, cache_sigmoid = self.activation.forward(X)
            self.cache["sigmoid" + str(i + 1)] = cache_sigmoid

            # Store the reg for the current W
            self.reg['W' + str(i + 1)] = np.sum(W ** 2) * self.reg_strength

        # last layer contains no activation functions
        W, b = self.params['W' + str(self.num_layer)],\
               self.params['b' + str(self.num_layer)]
        y, cache_affine = affine_forward(X, W, b)
        self.cache["affine" + str(self.num_layer)] = cache_affine
        self.reg['W' + str(self.num_layer)] = np.sum(W ** 2) * self.reg_strength

        return y

    def backward(self, dy):
        """
        Performs the backward pass of the model.

        :param dy: N x 1 array. The gradient wrt the output of the network.
        :return: Gradients of the model output wrt the model weights
        """

        # Note that last layer has no activation
        cache_affine = self.cache['affine' + str(self.num_layer)]
        dh, dW, db = affine_backward(dy, cache_affine)
        self.grads['W' + str(self.num_layer)] = \
            dW + 2 * self.reg_strength * self.params['W' + str(self.num_layer)]
        self.grads['b' + str(self.num_layer)] = db

        # The rest sandwich layers
        for i in range(self.num_layer - 2, -1, -1):
            # Unpack cache
            cache_sigmoid = self.cache['sigmoid' + str(i + 1)]
            cache_affine = self.cache['affine' + str(i + 1)]

            # Activation backward
            dh = self.activation.backward(dh, cache_sigmoid)

            # Affine backward
            dh, dW, db = affine_backward(dh, cache_affine)

            # Refresh the gradients
            self.grads['W' + str(i + 1)] = dW + 2 * self.reg_strength * \
                                           self.params['W' + str(i + 1)]
            self.grads['b' + str(i + 1)] = db

        return self.grads

    def save_model(self):
        self.eval()
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(model, open(directory + '/' + self.model_name + '.p', 'wb'))

    def get_dataset_prediction(self, loader):
        self.eval()
        scores = []
        labels = []
        
        for batch in loader:
            X = batch['image']
            y = batch['label']
            score = self.forward(X)
            scores.append(score)
            labels.append(y)
            
        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()

        return labels, preds, acc
    
    def eval(self):
        """sets the network in evaluation mode, i.e. only computes forward pass"""
        self.return_grad = False
        
        # Delete unnecessary caches, to mitigate a memory prolbem.
        self.reg = {}
        self.cache = {}
        
    def reset_weights(self):
        self.params = {'W1':self.std * np.random.randn(self.input_size, self.hidden_size),
                       'b1': np.zeros(self.hidden_size)}

        for i in range(self.num_layer - 2):
            self.params['W' + str(i + 2)] = self.std * np.random.randn(self.hidden_size,
                                                                  self.hidden_size)
            self.params['b' + str(i + 2)] = np.zeros(self.hidden_size)

        self.params['W' + str(self.num_layer)] = self.std * np.random.randn(self.hidden_size,
                                                                  self.num_classes)
        self.params['b' + str(self.num_layer)] = np.zeros(self.num_classes)

        self.grads = {}
        self.reg = {}
        for i in range(self.num_layer):
            self.grads['W' + str(i + 1)] = 0.0
            self.grads['b' + str(i + 1)] = 0.0
        
#########################################

class ReLU:
    def forward(self, x):
        self.cache = x
        return np.maximum(0, x), x

    def backward(self, dout, cache):
        dx = dout
        dx[cache <= 0] = 0
        return dx

class BatchNorm:
    def __init__(self, input_dim, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(input_dim)
        self.running_var = np.ones(input_dim)

    def forward(self, x, gamma, beta, mode='train'):
        if mode == 'train':
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            x_norm = (x - mu) / np.sqrt(var + self.eps)
            out = gamma * x_norm + beta
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            self.cache = (x, x_norm, mu, var, gamma, beta)
            return out, self.cache
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = gamma * x_norm + beta
            return out, None

    def backward(self, dout, cache):
        x, x_norm, mu, var, gamma, beta = cache
        N, D = x.shape
        
        dx_norm = dout * gamma
        dvar = np.sum(dx_norm * (x - mu) * -0.5 * (var + self.eps)**(-1.5), axis=0)
        dmu = np.sum(dx_norm * -1 / np.sqrt(var + self.eps), axis=0) + dvar * np.mean(-2 * (x - mu), axis=0)
        
        dx = (dx_norm / np.sqrt(var + self.eps)) + (dvar * 2 * (x - mu) / N) + (dmu / N)
        dgamma = np.sum(dout * x_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
        
        return dx, dgamma, dbeta


class MyOwnNetwork(Network):
    def __init__(self, activation=ReLU, num_layer=3, input_size=3 * 32 * 32, hidden_size=100, 
                 std=1e-3, num_classes=10, reg=0.0, dropout=0.5, **kwargs):
        super().__init__("cifar10_classification_net")
        
        self.activation = activation()
        self.reg_strength = reg
        self.dropout = dropout
        self.num_layer = num_layer
        self.std = std
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.reset_weights()
        
        # Initialize batch norm params
        self.bn_params = []
        for i in range(self.num_layer - 1):
            self.bn_params.append({'gamma': np.ones(hidden_size), 'beta': np.zeros(hidden_size)})
        
    def reset_weights(self):
        self.params = {'W1': np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size),
                       'b1': np.zeros(self.hidden_size)}
        
        for i in range(self.num_layer - 2):
            self.params['W' + str(i + 2)] = np.random.randn(self.hidden_size, self.hidden_size) * np.sqrt(2.0 / self.hidden_size)
            self.params['b' + str(i + 2)] = np.zeros(self.hidden_size)
        
        # Adjusting the output layer to match the number of classes
        self.params['W' + str(self.num_layer)] = np.random.randn(self.hidden_size, self.num_classes) * np.sqrt(2.0 / self.hidden_size)
        self.params['b' + str(self.num_layer)] = np.zeros(self.num_classes)
        
        self.grads = {}
        self.reg = {}
        for i in range(self.num_layer):
            self.grads['W' + str(i + 1)] = 0.0
            self.grads['b' + str(i + 1)] = 0.0

    def forward(self, X, mode='train'):
        self.cache = {}
        self.reg = {}
        X = X.reshape(X.shape[0], -1)
        
        for i in range(self.num_layer - 1):
            W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]
            
            X, cache_affine = affine_forward(X, W, b)
            self.cache['affine' + str(i + 1)] = cache_affine
            
            X, cache_bn = BatchNorm(X.shape[1]).forward(X, self.bn_params[i]['gamma'], self.bn_params[i]['beta'], mode)
            self.cache['bn' + str(i + 1)] = cache_bn
            
            X, cache_relu = self.activation.forward(X)
            self.cache['relu' + str(i + 1)] = cache_relu
            
            if mode == 'train' and self.dropout > 0:
                mask = (np.random.rand(*X.shape) < self.dropout) / self.dropout
                X *= mask
                self.cache['dropout_mask' + str(i + 1)] = mask
                
            self.reg['W' + str(i + 1)] = np.sum(W ** 2) * self.reg_strength
        
        W, b = self.params['W' + str(self.num_layer)], self.params['b' + str(self.num_layer)]
        y, cache_affine = affine_forward(X, W, b)
        self.cache['affine' + str(self.num_layer)] = cache_affine
        self.reg['W' + str(self.num_layer)] = np.sum(W ** 2) * self.reg_strength
        
        # Apply softmax activation for multi-class classification
        y -= np.max(y, axis=1, keepdims=True)  # for numerical stability
        probs = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return probs

    def backward(self, dy):
        cache_affine = self.cache['affine' + str(self.num_layer)]
        dh, dW, db = affine_backward(dy, cache_affine)
        self.grads['W' + str(self.num_layer)] = dW + 2 * self.reg_strength * self.params['W' + str(self.num_layer)]
        self.grads['b' + str(self.num_layer)] = db
        
        for i in range(self.num_layer - 2, -1, -1):
            if self.dropout > 0:
                mask = self.cache['dropout_mask' + str(i + 1)]
                dh *= mask
                
            cache_relu = self.cache['relu' + str(i + 1)]
            dh = self.activation.backward(dh, cache_relu)
            
            cache_bn = self.cache['bn' + str(i + 1)]
            dh, dgamma, dbeta = BatchNorm(dh.shape[1]).backward(dh, cache_bn)
            self.grads['gamma' + str(i + 1)] = dgamma
            self.grad
            
            
