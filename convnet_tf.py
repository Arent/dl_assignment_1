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


  def conv_layer(self, input_data, name_scope, out_channels, filter_shape=[5,5], activation_fn= tf.nn.relu,
                strides=[1,1,1,1], padding="SAME", pooling_stride=[2,2], pooling_kernel=[3,3]):
    #function that performs convolution, activation and max pooling
      in_shape = input_data.get_shape()

      batch_size=in_shape[0]
      in_channels= in_shape[3]
      print('inchannels', in_channels)
      weight_shape = filter_shape +[in_channels, out_channels]
      bias_shape = [float(in_shape[1])/pooling_stride[0], float(in_shape[2])/pooling_stride[1],out_channels ]
      with tf.variable_scope(name_scope):
        filter_weights = tf.get_variable("weights", shape=weight_shape)
        biases = tf.get_variable("biases", initializer=tf.zeros(bias_shape)) #TODO MAKE THIS DYNAMIC
        
        output = tf.nn.conv2d(
                      input=input_data,
                      filter=filter_weights,
                      strides=strides,
                      padding=padding,
                      use_cudnn_on_gpu=False,
                      name='convolution')
        if activation_fn is not None:
          output = activation_fn(output + biases)

        output_layer1 =tf.nn.max_pool(
                      value=output,
                      ksize=[1] + pooling_kernel + [1],
                      strides=[1] + pooling_stride + [1],
                      padding="SAME" ,
                      name='Maxpooling')


  def fully_connected_layer(self, name_scope, input_data,output_dimensions,  activation_fn=tf.nn.relu, dropout_rate=0 ):
    # This function creates a fully connected layer.
    in_shape = input_data.get_shape()
    with tf.variable_scope(name_scope):
      weights = tf.get_variable("weights", shape=[in_shape[1], output_dimensions])
      biases = tf.get_variable("biases", shape=[1,output_dimensions])
      output = tf.matmul(x, weights) + biases
      if activation_fn is not None:
        output = activation_fn(output)
      output = tf.nn.dropout(activated_output, 1.0 - dropout_rate) 
    return output

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
    output_conv1 = self.conv_layer(input_data=x, name_scope='conv1', out_channels=64)
    output_conv2 = self.conv_layer(input_data=output_conv1, name_scope='conv1', out_channels=64)
    input_fc = tf.reshape(output_conv2,[batch_size,-1])
    output_fc1 = self.fully_connected_layer(name_scope="fc1", input_data=input_fc,output_dimensions=384,
                                            activation_fn=tf.nn.relu, dropout_rate=0)
    output_fc2 = self.fully_connected_layer(name_scope="fc2", input_data=output_fc1,output_dimensions=192,
                                            activation_fn=tf.nn.relu, dropout_rate=0)
    logits= self.fully_connected_layer(name_scope="fc3", input_data=output_fc2,output_dimensions=10,
                                            activation_fn=None, dropout_rate=0)

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

