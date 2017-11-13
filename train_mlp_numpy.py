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
import matplotlib.pyplot as plt
import pickle

# Default constants
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.001
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 2500
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

  plot_data= {}

  for layer in [100, 150, 200]:
    weight_decay = {}
    for weight_decay in [0, 0.001, 0.005, 0.01, 0.02]:
      weight_decay_dict = {}
      print("weight decay:", weight_decay)
      model = MLP(n_hidden=dnn_hidden_units,n_classes=10,batch_size=FLAGS.batch_size, input_dim=32*32*3, 
                weight_decay=FLAGS.weight_reg_strength, weight_scale=FLAGS.weight_init_scale)

      Datasets = utils.get_cifar10(data_dir = DATA_DIR_DEFAULT, one_hot = True, validation_size = 0)
      
      train_losses = []
      test_losses = []
      for i in range(FLAGS.max_steps): 
        train_batch, train_labels = Datasets.train.next_batch(batch_size = FLAGS.batch_size)
        #Get the model output
        train_data = train_batch.reshape([FLAGS.batch_size,-1])/10.0
        logits = model.inference(x=train_data)
        #Get the loss and let the model set the loss derivative.
        loss = model.loss(logits=logits, labels=train_labels)
        train_losses.append(loss)

        #Perform training step
        model.train_step(loss=loss, flags=FLAGS)

        #Every 100th iteratin print accuracy on the whole test set.
        if i % 100 == 0:
          # for layer in model.layers:

          test_batch, test_labels = Datasets.test.next_batch(batch_size = Datasets.test.num_examples) #
          test_data = test_batch.reshape([Datasets.test.num_examples,-1])/10.0
          logits = model.inference(x=test_data )

          loss = model.loss(logits=logits, labels=test_labels)
          test_losses.append(loss)
          print('-- Step: ', i, " accuracy: ",model.accuracy(logits=logits,labels=test_labels),'loss', loss )

      weight_decay_dict["train_losses"] = train_losses
      weight_decay_dict["test_losses"] = test_losses
    plot_data[weight_decay] = weight_decay  

  pickle.dump( plot_data, open( "plot_data_mlp_num_norm.p", "wb" ) )
    # plt.figure()
    # plt.plot(train_losses)
    # plt.title("Cross entropy loss training data, weight decay: " + str(weight_decay))
    # plt.savefig("train_losses" + str(weight_decay).replace(".",",") +".png" )
    # plt.figure()
    # plt.title("Cross entropy loss test data, weight decay: " + str(weight_decay) )
    # plt.plot(test_losses)
    # plt.savefig("test_losses"+ str(weight_decay).replace(".",",")  +".png")

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
