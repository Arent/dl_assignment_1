"""
This module implements training and evaluation of a multi-layer perceptron in TensorFlow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
from mlp_tf import MLP
import cifar10_utils as utils

# Default constants
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.001
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
DROPOUT_RATE_DEFAULT = 0.
DNN_HIDDEN_UNITS_DEFAULT = '100'
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
ACTIVATION_DEFAULT = 'relu'
OPTIMIZER_DEFAULT = 'sgd'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
# Directory for tensorflow logs
LOG_DIR_DEFAULT = './logs/cifar10'


# This is the list of options for command line arguments specified below using argparse.
# Make sure that all these options are available so we can automatically test your code
# through command line arguments.

# You can check the TensorFlow API at
# https://www.tensorflow.org/programmers_guide/variables
# https://www.tensorflow.org/api_guides/python/contrib.layers#Initializers
WEIGHT_INITIALIZATION_DICT = {'xavier': tf.contrib.layers.xavier_initializer, # Xavier initialisation
                              'normal': tf.random_normal, # Initialization from a standard normal
                              'uniform': tf.random_uniform, # Initialization from a uniform distribution
                             }

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/contrib.layers#Regularizers
WEIGHT_REGULARIZER_DICT = {'none': None, # No regularization
                           'l1': tf.contrib.layers.l1_regularizer, # L1 regularization
                           'l2': tf.contrib.layers.l2_regularizer # L2 regularization
                          }
# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/nn

ACTIVATION_DICT = {'relu': tf.nn.relu, # ReLU
                   'elu': tf.nn.elu, # ELU
                   'tanh': tf.tanh, #Tanh
                   'sigmoid': tf.sigmoid} #Sigmoid

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/train
OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer, # Gradient Descent
                  'adadelta': tf.train.AdagradDAOptimizer, # Adadelta
                  'adagrad': tf.train.AdagradOptimizer, # Adagrad
                  'adam': tf.train.AdamOptimizer, # Adam
                  'rmsprop': tf.train.RMSPropOptimizer # RMSprop
                  }

FLAGS = None

def train():
  """
  Performs training and evaluation of MLP model. Evaluate your model each 100 iterations
  as you did in the task 1 of this assignment. 
  """
  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  tf.set_random_seed(42)
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  #Add parguments to the initializer
  if FLAGS.weight_init == 'xavier':
    initializer = WEIGHT_INITIALIZATION_DICT['xavier']
  elif FLAGS.weight_init == 'uniform':
    initializer = WEIGHT_INITIALIZATION_DICT['uniform'](minval=-FLAGS.weight_init_scale, maxval=FLAGS.weight_init_scale)
  elif FLAGS.weight_init == 'normal':
    initializer = WEIGHT_INITIALIZATION_DICT['xavier']()

    # initializer = WEIGHT_INITIALIZATION_DICT['normal']


  if FLAGS.weight_reg:
    regularizer = WEIGHT_REGULARIZER_DICT[FLAGS.weight_reg](FLAGS.weight_reg_strength)

  else:
    regularizer = None

  activation_fn=ACTIVATION_DICT[FLAGS.activation]
  optimizer=OPTIMIZER_DICT[FLAGS.optimizer](FLAGS.learning_rate)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  model =  MLP(n_hidden=dnn_hidden_units, n_classes=10, is_training=True,input_dim=32*32*3,
               activation_fn = activation_fn, dropout_rate = 0.5)

  x = tf.placeholder(dtype=tf.float32)
  labels = tf.placeholder(dtype=tf.float32)

  logits = model.inference(x)
  loss = model.loss(logits=logits, labels=labels)
  train_op = model.train_step(loss=loss, flags=FLAGS)
  accuracy = model.accuracy(logits=logits, labels=labels)

  Datasets = utils.get_cifar10(data_dir = DATA_DIR_DEFAULT, one_hot = True, validation_size = 0)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    for i in range(FLAGS.max_steps): 
      train_batch = Datasets.train.next_batch(batch_size = FLAGS.batch_size)
      train_data = train_batch[0].reshape(FLAGS.batch_size,-1)
      train_labels = train_batch[1]
      #Get the model output
      #Perform training step
      t, loss_e = sess.run([train_op, loss], feed_dict={x:train_data, labels:train_labels })
      # print('step: ', i, 'training_loss:', loss_e)
      #Every 100th iteratin print accuracy on the whole test set.
      if i % 100 == 0:
        # for layer in model.layers:
        test_batch = Datasets.test.next_batch(batch_size = 10000) 
        test_data = test_batch[0].reshape([10000,-1])
        test_labels = test_batch[1]
        accuracy_e, loss_e = sess.run([accuracy, loss],feed_dict={x:test_data,labels:test_labels } )
        print('-- Step: ', i, " accuracy: ",accuracy_e,'loss', loss_e )
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main(_):
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  # Make directories if they do not exists yet
  if not tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.MakeDirs(FLAGS.log_dir)
  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MakeDirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  
  parser.add_argument('--weight_init', type = str, default = WEIGHT_INITIALIZATION_DEFAULT,
                      help='Weight initialization type [xavier, normal, uniform].')
  
  parser.add_argument('--weight_init_scale', type = float, default = WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                      help='Weight initialization scale (e.g. std of a Gaussian).')
  
  parser.add_argument('--weight_reg', type = str, default = WEIGHT_REGULARIZER_DEFAULT,
                      help='Regularizer type for weights of fully-connected layers [none, l1, l2].')
  
  parser.add_argument('--weight_reg_strength', type = float, default = WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                      help='Regularizer strength for weights of fully-connected layers.')
  
  parser.add_argument('--dropout_rate', type = float, default = DROPOUT_RATE_DEFAULT,
                      help='Dropout rate.')
  
  parser.add_argument('--activation', type = str, default = ACTIVATION_DEFAULT,
                      help='Activation function [relu, elu, tanh, sigmoid].')
  
  parser.add_argument('--optimizer', type = str, default = OPTIMIZER_DEFAULT,
                      help='Optimizer to use [sgd, adadelta, adagrad, adam, rmsprop].')
  
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  
  parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
  
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run()
