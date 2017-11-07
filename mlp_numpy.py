"""
This module implements a multi-layer perceptron in NumPy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform inference, training and it
  can also be used for evaluating prediction performance.
  """

  def __init__(self, n_hidden, n_classes,batch_size =100, input_dim=32*32*3, weight_decay=0.0, weight_scale=0.0001):
    """
    Constructor for an MLP object. Default values should be used as hints for
    the usage of each parameter. Weights of the linear layers should be initialized
    using normal distribution with mean = 0 and std = weight_scale. Biases should be
    initialized with constant 0. All activation functions are ReLUs.

    Args:
      n_hidden: list of ints, specifies the number of units
                     in each hidden layer. If the list is empty, the MLP
                     will not have any hidden units, and the model
                     will simply perform a multinomial logistic regression.
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the MLP.
      weight_decay: L2 regularization parameter for the weights of linear layers.
      weight_scale: scale of normal distribution to initialize weights.

    """
    self.input_dim= input_dim
    self.batch_size = batch_size
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.weight_decay = weight_decay
    self.weight_scale = weight_scale

    # initialize the weights
    

    self.layers = []
    for  size_in, size_out in zip([self. input_dim]+ self.n_hidden , self.n_hidden + [self.n_classes]):
      layer = {}
      layer["weights"] = np.random.normal(0, weight_scale, [size_in, size_out]) 
      layer["bias"]  = np.zeros([batch_size, size_out])
      layer["activation_function"] = self.relu 
      layer["size"] = size_out

      self.layers.append(layer)

    #the last layer doesnt have an activation funciton
    self.layers[-1]["activation_function"] = self.softmax




  def inference(self, x):
    """
    Performs inference given an input array. This is the central portion
    of the network. Here an input tensor is transformed through application
    of several hidden layer transformations (as defined in the constructor).
    We recommend you to iterate through the list self.n_hidden in order to
    perform the sequential transformations in the MLP. Do not forget to
    add a linear output layer (without non-linearity) as the last transformation.

    It can be useful to save some intermediate results for easier computation of
    gradients for backpropagation during training.
    Args:
      x: 2D float array of size [batch_size, input_dimensions]

    Returns:
      logits: 2D float array of size [batch_size, self.n_classes]. Returns
             the logits outputs (before softmax transformation) of the
             network. These logits can then be used with loss and accuracy
             to evaluate the model.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.input = x

    self.layers[0]["output"] = np.dot(x, self.layers[0]["weights"]) + self.layers[0]["bias"]
    f = self.layers[0]["activation_function"]
    self.layers[0]["activated_output"] = f(self.layers[0]["output"])

    for layer_i_1, layer_i in zip(self.layers, self.layers[1:]):
      layer_i["output"] = np.dot(layer_i_1["output"], layer_i["weights"])  + layer_i["bias"]
      f = layer_i["activation_function"]
      layer_i["activated_output"] = f(layer_i["output"])

    logits = self.layers[-1]["activated_output"]

    ########################
    # END OF YOUR CODE    #
    #######################

    return logits

  def loss(self, logits, labels):
    """
    Computes the multiclass cross-entropy loss from the logits predictions and
    the ground truth labels. The function will also add the regularization
    loss from network weights to the total loss that is return.

    It can be useful to compute gradients of the loss for an easier computation of
    gradients for backpropagation during training.

    Args:
      logits: 2D float array of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int array of size [batch_size, self.n_classes]
                   with one-hot encoding. Ground truth labels for each
                   sample in the batch.
    Returns:
      loss: scalar float, full loss = cross_entropy + reg_loss
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #This approach uses numpy tot slice the logits data into probabilities for examples 
    #with label 1 and with label 0, then the cross entropy is calculated for both. 
    # another aprouch would be: -np.sum(np.log((labels + labels -1) * logits1  + (1-labels)))
    # Which is only marginally faster
    loss_label_1 = -np.sum(np.log(np.maximum(1- logits.flatten()[labels.flatten() ==1], 10**(-10)))) 
    # print('loss_label_1',loss_label_1)
    loss_label_0 = -np.sum(np.log(np.maximum(1- logits.flatten()[labels.flatten() ==0], 10**(-10)))) 
    loss = (loss_label_1 + loss_label_0)/logits.shape[0]
    # print('loss_label_0',loss_label_0)
    self.loss_derivative = logits - labels

      ########################
    # END OF YOUR CODE    #
    #######################

    return loss

  def train_step(self, loss, flags):
    """
    Implements a training step using a parameters in flags.
    Use Stochastic Gradient Descent to update the parameters of the MLP.

    Args:
      loss: scalar float Tensor.
      flags: contains necessary parameters for optimization.
    Returns:

    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################


    #Calculate deltas
    #The delta of the last layer is the derivative of the loss function
    self.layers[-1]["delta"] = self.loss_derivative
    #Loop over the layers in reversed order,  layer_i_1 = layer_i-1, layer_= layer i
    for layer_i_1, layer_i in zip(reversed(self.layers[0:-1]), reversed(self.layers)):
      f = layer_i_1["activation_function"]
      layer_i_1["delta"] = layer_i["delta"].dot(layer_i["weights"].T) * f(layer_i_1["output"],derivative=True)
    

    #Calculate derivatives for the weights and bias and apply them
    #The 'activated output' of the first layer is just the input
    self.layers[0]["weights_derivative"]  = self.input.T.dot(self.layers[0]["delta"]) 
    # loop over the layers, layer_i_1 = layer i-1, layer_i =layer i
    for layer_i_1, layer_i in zip(self.layers, self.layers[1:]):
      layer_i["weights_derivative"]  = layer_i_1["activated_output"].T.dot(layer_i['delta']) 
      layer_i["weights"] -= flags.learning_rate*(layer_i["weights_derivative"]/flags.batch_size)
      layer_i["bias"] -= flags.learning_rate*(layer_i["delta"]/flags.batch_size)

    

    ########################
    # END OF YOUR CODE    #
    #######################

    return

  def accuracy(self, logits, labels):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      logits: 2D float array of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int array of size [batch_size, self.n_classes]
                 with one-hot encoding. Ground truth labels for
                 each sample in the batch.
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # accuracy = np.mean(labels - (logits>0.))
    accuracy = np.sum(np.argmax(logits, axis=1) == np.argmax(labels, axis=1)) / labels.shape[0]
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy

  def relu(self, x, derivative=False):
    if derivative:
      return (x > 0) * 1 + (x <= 0) * 0
    else:
      return x * (x > 0)
  
  #convenience function
  def lin(self, x, derivative=False):
    return x

  def softmax(self, x):
    e_x = np.exp(x - np.max(x,axis=1)[:,np.newaxis])
    return e_x / np.sum(e_x,axis=1)[:,np.newaxis]




