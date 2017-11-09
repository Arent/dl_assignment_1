"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
import cifar10_utils as utils

# Default constants
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.001
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
DNN_HIDDEN_UNITS_DEFAULT = '100'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = 'cifar10/cifar-10-batches-py'

FLAGS = None

def train():
  """
  Performs training and evaluation of MLP model. Evaluate your model on the whole test set each 100 iterations.
  """
  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  model = MLP(n_hidden=dnn_hidden_units,n_classes=10,batch_size=FLAGS.batch_size, input_dim=32*32*3, 
            weight_decay=FLAGS.weight_reg_strength, weight_scale=FLAGS.weight_init_scale)

  Datasets = utils.get_cifar10(data_dir = DATA_DIR_DEFAULT, one_hot = True, validation_size = 0)
  
  for i in range(1500): #(FLAGS.max_steps):
    train_batch = Datasets.train.next_batch(batch_size = FLAGS.batch_size)
    #Get the model output
    logits = model.inference(x=train_batch[0].reshape([FLAGS.batch_size,32*32*3]))
    #Get the loss and let the model set the loss derivative.
    loss = model.loss(logits=logits, labels=train_batch[1])
    #Perform training step
    model.train_step(loss=loss, flags=FLAGS)

    #Every 100th iteratin print accuracy on the whole test set.
    if i % 100 == 0:
      # for layer in model.layers:
      test_batch = Datasets.test.next_batch(batch_size = 200) #Datasets.test.num_examples
      logits = model.inference(x=test_batch[0].reshape([200,32*32*3]))
      print('-- Step: ', i, " accuracy: ",model.accuracy(logits=logits,labels=test_batch[1]),'loss', loss )

  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

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
  parser.add_argument('--weight_init_scale', type = float, default = WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                      help='Weight initialization scale (e.g. std of a Gaussian).')
  parser.add_argument('--weight_reg_strength', type = float, default = WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                      help='Regularizer strength for weights of fully-connected layers.')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
