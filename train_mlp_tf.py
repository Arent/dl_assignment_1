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
MAX_STEPS_DEFAULT = 2501 #2501
DROPOUT_RATE_DEFAULT = 0.5
DNN_HIDDEN_UNITS_DEFAULT = '350'
WEIGHT_INITIALIZATION_DEFAULT = 'xavier'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
ACTIVATION_DEFAULT = 'relu'
OPTIMIZER_DEFAULT = 'adam'

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
    print('dnn_hidden_units',dnn_hidden_units )
  else:
    dnn_hidden_units = []

  #Add parguments to the initializer
  if FLAGS.weight_init == 'normal':
    initializer = WEIGHT_INITIALIZATION_DICT['normal']
    initializer_arg = FLAGS.weight_init_scale
  elif FLAGS.weight_init == 'uniform':
    initializer = WEIGHT_INITIALIZATION_DICT['uniform']#() #(minval=-FLAGS.weight_init_scale, maxval=FLAGS.weight_init_scale)
    initializer_stddev = None
  elif FLAGS.weight_init == 'xavier':
    initializer = WEIGHT_INITIALIZATION_DICT['xavier']()
    initializer_arg = None

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
  for layer in [100, 250]:
    for dropout in [0.0, 0.25, 0.5, 0.75]: #, 0.5]:
      for reg in [0.001, 0.01, 0.1]:#,0.01, 0.1, 0.5]: 

        train_logs_path = 'logs/cifar10/mlp_tf/train_scaled/layersize'+str(layer) + '_dr_' + str(dropout) + '_reg_' + str(reg) + '/'
        test_logs_path = 'logs/cifar10/mlp_tf/test_scaled/layersize'+str(layer) + '_dr_' + str(dropout) + '_reg_' + str(reg) + '/'

        print(train_logs_path)
        if not tf.gfile.Exists(train_logs_path):
          tf.gfile.MakeDirs(train_logs_path)
        if not tf.gfile.Exists(test_logs_path):
          tf.gfile.MakeDirs(test_logs_path)

        regularizer = WEIGHT_REGULARIZER_DICT[FLAGS.weight_reg](reg)

        model =  MLP(n_hidden=[layer], n_classes=10, is_training=True,input_dim=32*32*3,
                     activation_fn = activation_fn, weight_initializer = initializer, initializer_stddev=None,
                     weight_regularizer = regularizer,
                     optimizer= optimizer)

        x = tf.placeholder(dtype=tf.float32)
        labels = tf.placeholder(dtype=tf.float32)
        dropout_rate = tf.placeholder(dtype=tf.float32)

        logits = model.inference(x, dropout_rate)
        loss = model.loss(logits=logits, labels=labels)
        train_op = model.train_step(loss=loss, flags=FLAGS)
        accuracy = model.accuracy(logits=logits, labels=labels)

        Datasets = utils.get_cifar10(data_dir = DATA_DIR_DEFAULT, one_hot = True, validation_size = 0)
        train_merged_summary_op = tf.summary.merge_all(key='train')
        test_merged_summary_op = tf.summary.merge_all(key='test')
        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          train_summary_writer = tf.summary.FileWriter(train_logs_path, graph=tf.get_default_graph())
          test_summary_writer = tf.summary.FileWriter(test_logs_path, graph=tf.get_default_graph())


          for i in range(FLAGS.max_steps): 
            train_batch, train_labels = Datasets.train.next_batch(batch_size = FLAGS.batch_size)
            train_data = train_batch.reshape(FLAGS.batch_size,-1) / 255.0
            #Get the model output
            #Perform training step
            ac, t, loss_e, summary = sess.run([accuracy, train_op, loss, train_merged_summary_op], feed_dict={x:train_data, labels:train_labels, dropout_rate:dropout })

            # Write logs at every iteration, only loss and accuracy
            train_summary_writer.add_summary(summary, i)

            #Every 100th iteratin print accuracy on the whole test set.
            if i % 100 == 0:
              # for layer in model.layers:
              test_batch,  test_labels = Datasets.test.next_batch(batch_size = 10000) 
              test_data = test_batch.reshape([10000,-1]) / 255.0
              accuracy_e, loss_e, summary = sess.run([accuracy, loss, test_merged_summary_op],feed_dict={x:test_data,labels:test_labels, dropout_rate:0.0 } )
              test_summary_writer.add_summary(summary, i)

              print('-- Step: ', i, " accuracy: ",accuracy_e,'loss', loss_e )
        tf.reset_default_graph()
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
