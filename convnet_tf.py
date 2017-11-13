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

  def __init__(self, n_classes = 10, batch_norm=False, residual=False, dropout_rate=0.0, use_gpu=False):
    """
    Constructor for an ConvNet object. Default values should be used as hints for
    the usage of each parameter.
    Args:
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the ConvNet.
    """
    self.batch_norm = batch_norm
    self.residual = residual
    self.n_classes = n_classes
    self.dropout_rate = dropout_rate
    self.use_gpu = use_gpu

  def conv_layer(self, input_data, name_scope, in_channels, out_channels, output_shape, filter_shape=[5,5], activation_fn= tf.nn.relu,
                strides=[1,1,1,1], padding="SAME", pooling_stride=[2,2], pooling_kernel=[3,3], batch_norm=False, max_pool=True):
    #function that performs convolution, activation and max pooling
    weight_shape = filter_shape +[in_channels, out_channels]
    bias_shape = output_shape + [out_channels ]

    with tf.variable_scope(name_scope):
      filter_weights = tf.get_variable("weights", shape=weight_shape)
      biases = tf.get_variable("biases", initializer=tf.zeros(bias_shape)) #TODO MAKE THIS DYNAMIC
      
      output = tf.nn.conv2d(
                    input=input_data,
                    filter=filter_weights,
                    strides=strides,
                    padding=padding,
                    use_cudnn_on_gpu=self.use_gpu,
                    name='convolution')

 
      if activation_fn is not None:
        output = activation_fn(output + biases)

      if max_pool:
        output =tf.nn.max_pool(
                    value=output,
                    ksize=[1] + pooling_kernel + [1],
                    strides=[1] + pooling_stride + [1],
                    padding="SAME" ,
                    name='Maxpooling')
      if batch_norm:
        output= tf.layers.batch_normalization(output)
    return output


  def fully_connected_layer(self, name_scope, input_data,input_dimensions, output_dimensions,  activation_fn=tf.nn.relu, dropout_rate=0, batch_norm=True):
    # This function creates a fully connected layer.
    with tf.variable_scope(name_scope):
      weights = tf.get_variable("weights", shape=[input_dimensions, output_dimensions])
      biases = tf.get_variable("biases", shape=[1,output_dimensions])
      input_data = tf.nn.dropout(input_data, 1.0 - dropout_rate) 
      output = tf.matmul(input_data, weights) + biases
      if activation_fn is not None:
        output = activation_fn(output)
      if batch_norm:
        output= tf.layers.batch_normalization(output)
      

    return output

  def resUnit(self, input_data, name_scope, in_channels, out_channels, output_shape):
    with tf.variable_scope(name_scope):
      input_data = tf.layers.batch_normalization(input_data)
      activated_input = tf.nn.relu(input_data)
      conv1= self.conv_layer(input_data=activated_input, name_scope='conv1', 
                  in_channels=in_channels, out_channels=out_channels, 
                  output_shape=output_shape, batch_norm=False, activation_fn=None, max_pool=False)

      conv1_norm = tf.layers.batch_normalization(conv1)
      conv1_norm_activated = tf.nn.relu(conv1_norm)

      conv2 = self.conv_layer(input_data=conv1_norm_activated, name_scope='conv2', 
                  in_channels=out_channels, out_channels=out_channels, 
                  output_shape=output_shape, batch_norm=False, activation_fn=None, max_pool=False)
      
      return input_data + conv2

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


    if self.residual:
      #First add a regular convolution to ensure depth channels =64
      output_conv1 = self.conv_layer(input_data=x, name_scope='conv1', 
        in_channels=3, out_channels=64, output_shape=[32,32], max_pool=False)

      # RES -  RES -> MAX
      output_res_1 = self.resUnit( input_data=output_conv1, name_scope='res1', in_channels=64, out_channels=64, output_shape=[32,32])
      output_res_2 = self.resUnit( input_data=output_res_1, name_scope='res2', in_channels=64, out_channels=64, output_shape=[32,32])
      output_res_2_m = tf.nn.max_pool(value=output_res_2, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME" , name='Maxpooling1')

      # RES -  RES -> MAX
      output_res_3 = self.resUnit( input_data=output_res_2_m, name_scope='res3', in_channels=64, out_channels=64, output_shape=[16,16])
      output_res_4 = self.resUnit( input_data=output_res_3, name_scope='res4', in_channels=64, out_channels=64, output_shape=[16,16])
      output = tf.nn.max_pool(value=output_res_4, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME" , name='Maxpooling2')
      # output_res_5 = self.resUnit( input_data=output_res_4_m, name_scope='res5', in_channels=64, out_channels=64, output_shape=[16,16])
      # output_res_6 = self.resUnit( input_data=output_res_5, name_scope='res6', in_channels=64, out_channels=64, output_shape=[16,16])
      # output = tf.nn.max_pool(value=output_res_6, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME" , name='Maxpooling3')


    else:
      output_conv1 = self.conv_layer(input_data=x, name_scope='conv1', 
        in_channels=3, out_channels=64, output_shape=[32,32])

      output_conv2 = self.conv_layer(input_data=output_conv1, name_scope='conv2',
        in_channels=64, out_channels=64, output_shape=[16,16])
      output = output_conv2

    
    batch_size = tf.shape(x)[0]
    input_fc = tf.reshape(output, [batch_size,-1])

    output_fc1 = self.fully_connected_layer(name_scope="fc1", input_data=input_fc,input_dimensions=8*8*64, output_dimensions=384,
                                            activation_fn=tf.nn.relu, dropout_rate=self.dropout_rate, batch_norm=self.batch_norm)
    output_fc2 = self.fully_connected_layer(name_scope="fc2", input_data=output_fc1,input_dimensions=384, output_dimensions=192,
                                            activation_fn=tf.nn.relu, dropout_rate=self.dropout_rate)
    logits= self.fully_connected_layer(name_scope="fc3", input_data=output_fc2, input_dimensions=192, output_dimensions=10,
                                            activation_fn=None, dropout_rate=self.dropout_rate, batch_norm=False)

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

