"""
This module implements a convolutional neural network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class ConvNet(object):
  """
  This class implements a convolutional neural network in TensorFlow.
  It incorporates a certain graph model to be trained and to be used
  in inference.
  """

  def __init__(self, n_classes = 10):
    """
    Constructor for an ConvNet object. Default values should be used as hints for
    the usage of each parameter.
    Args:
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the ConvNet.
    """
    self.n_classes = n_classes

  def inference(self, x):
    """
    Performs inference given an input tensor. This is the central portion
    of the network where we describe the computation graph. Here an input
    tensor undergoes a series of convolution, pooling and nonlinear operations
    as defined in this method. For the details of the model, please
    see assignment file.

    Here we recommend you to consider using variable and name scopes in order
    to make your graph more intelligible for later references in TensorBoard
    and so on. You can define a name scope for the whole model or for each
    operator group (e.g. conv+pool+relu) individually to group them by name.
    Variable scopes are essential components in TensorFlow for parameter sharing.
    Although the model(s) which are within the scope of this class do not require
    parameter sharing it is a good practice to use variable scope to encapsulate
    model.

    Args:
      x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]

    Returns:
      logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
              the logits outputs (before softmax transformation) of the
              network. These logits can then be used with loss and accuracy
              to evaluate the model.
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    batch_size = tf.shape(x)[0]
    with tf.variable_scope("conv1"): 
      filter_weights = tf.get_variable("weights", shape=[5, 5, 3, 64])
      print("TF.SHAPE(X)",tf.shape(x))
      biases = tf.get_variable("biases", initializer=tf.zeros([32, 32, 64])) #TODO MAKE THIS DYNAMIC
      output = tf.nn.conv2d(
                    input=x,
                    filter=filter_weights,
                    strides=[1,1,1,1],
                    padding="SAME",
                    use_cudnn_on_gpu=False,
                    name='convolution')
      activated_output = tf.nn.relu(output + biases)

      output_layer1 =tf.nn.max_pool(
                    value=activated_output,
                    ksize=[1,3,3,1],
                    strides=[1,2,2,1],
                    padding="SAME",
                    name='Maxpooling')

    with tf.variable_scope("conv2"): 
      filter_weights = tf.get_variable("weights", shape=[5, 5, 64, 64])
      biases = tf.get_variable("biases", initializer=tf.zeros([16, 16, 64]))
      output = tf.nn.conv2d(
                    input=output_layer1,
                    filter=filter_weights,
                    strides=[1,1,1,1],
                    padding="SAME",
                    use_cudnn_on_gpu=False,
                    name='convolution')
      activated_output = tf.nn.relu(output + biases)

      output_layer2 =tf.nn.max_pool(
                    value=activated_output,
                    ksize=[1,3,3,1],
                    strides=[1,2,2,1],
                    padding="SAME",
                    name='Maxpooling')

    with tf.variable_scope("fc1"):
      input_fc = tf.reshape(output_layer2,[batch_size,-1])
      dimensions = tf.shape(input_fc)[1]
      weights = tf.get_variable("weights",shape=[8*8*64, 384] )
      biases = tf.get_variable("biases", initializer=tf.zeros([1,384]))
      output = tf.matmul(input_fc, weights) + biases
      activated_output = tf.nn.relu(output)

    with tf.variable_scope("fc2"):
      weights = tf.get_variable("weights",shape=[384, 192] )
      biases = tf.get_variable("biases", initializer=tf.zeros([1,192]))
      output = tf.matmul(activated_output, weights) + biases
      activated_output = tf.nn.relu(output)

    with tf.variable_scope("fc3"):
      weights = tf.get_variable("weights",shape=[192, 10] )
      biases = tf.get_variable("biases", initializer=tf.zeros([1,10]))
      output = tf.matmul(activated_output, weights) + biases
      logits = output    

      ########################
    # END OF YOUR CODE    #
    ########################
    return logits

  def loss(self, logits, labels):
    """
    Calculates the multiclass cross-entropy loss from the logits predictions and
    the ground truth labels. The function will also add the regularization
    loss from network weights to the total loss that is return.

    In order to implement this function you should have a look at
    tf.nn.softmax_cross_entropy_with_logits.
    
    You can use tf.summary.scalar to save scalar summaries of
    cross-entropy loss, regularization loss, and full loss (both summed)
    for use with TensorBoard. This will be useful for compiling your report.

    Args:
      logits: 2D float Tensor of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int Tensor of size [batch_size, self.n_classes]
                   with one-hot encoding. Ground truth labels for each
                   sample in the batch.

    Returns:
      loss: scalar float Tensor, full loss = cross_entropy + reg_loss
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    ########################
    # END OF YOUR CODE    #
    ########################

    return loss

  def train_step(self, loss, flags):
    """
    Implements a training step using a parameters in flags.

    Args:
      loss: scalar float Tensor.
      flags: contains necessary parameters for optimization.
    Returns:
      train_step: TensorFlow operation to perform one training step
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    optimizer = tf.train.AdamOptimizer(flags.learning_rate)
    train_step = optimizer.minimize(loss)
    ########################
    # END OF YOUR CODE    #
    #######################

    return train_step

  def accuracy(self, logits, labels):
    """
    Calculate the prediction accuracy, i.e. the average correct predictions
    of the network.
    As in self.loss above, you can use tf.scalar_summary to save
    scalar summaries of accuracy for later use with the TensorBoard.

    Args:
      logits: 2D float Tensor of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int Tensor of size [batch_size, self.n_classes]
                 with one-hot encoding. Ground truth labels for
                 each sample in the batch.

    Returns:
      accuracy: scalar float Tensor, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch.
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    correct   = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy  = tf.reduce_mean(tf.cast(correct, tf.float32))
    ########################
    # END OF YOUR CODE    #
    ########################

    return accuracy

