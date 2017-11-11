from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
from convnet_tf import ConvNet
import cifar10_utils as utils


LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32 ## TODO CHAGE TO 128
MAX_STEPS_DEFAULT = 201 # TODO CHANGE TO 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'


def train():
  """
  Performs training and evaluation of ConvNet model.

  First define your graph using class ConvNet and its methods. Then define
  necessary operations such as savers and summarizers. Finally, initialize
  your model within a tf.Session and do the training.

  ---------------------------
  How to evaluate your model:
  ---------------------------
  Evaluation on test set should be conducted over full batch, i.e. 10k images,
  while it is alright to do it over minibatch for train set.

  ---------------------------------
  How often to evaluate your model:
  ---------------------------------
  - on training set every print_freq iterations
  - on test set every eval_freq iterations

  ------------------------
  Additional requirements:
  ------------------------
  Also you are supposed to take snapshots of your model state (i.e. graph,
  weights and etc.) every checkpoint_freq iterations. For this, you should
  study TensorFlow's tf.train.Saver class.
  """

  # Set the random seeds for reproducibility. DO NOT CHANGE.
  tf.set_random_seed(42)
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  ########################
  model =  ConvNet( n_classes=10)

  x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
  labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])

  logits = model.inference(x)
  loss = model.loss(logits=logits, labels=labels)
  train_op = model.train_step(loss=loss, flags=FLAGS)
  accuracy = model.accuracy(logits=logits, labels=labels)

  cifar_10 = utils.get_cifar10(data_dir = DATA_DIR_DEFAULT, one_hot = True, validation_size = 0)
  test_data, test_labels = cifar_10.test.next_batch(batch_size = 1000) 


  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    for i in range(FLAGS.max_steps): 
      train_data, train_labels = cifar_10.train.next_batch(batch_size = FLAGS.batch_size)
      #Perform training step
      t, loss_e = sess.run([train_op, loss], feed_dict={x:train_data, labels:train_labels })

      #Every 100th iteratin print accuracy on the whole test set.
      if i % 100 == 0:
        accuracy_e, loss_e = sess.run([accuracy, loss],feed_dict={x:test_data,labels:test_labels } )
        print('-- Step: ', i, " accuracy: ",accuracy_e,'loss', loss_e )
  ########################  ########################
  # END OF YOUR CODE    #
  ########################

def initialize_folders():
  """
  Initializes all folders in FLAGS variable.
  """

  if not tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.MakeDirs(FLAGS.log_dir)

  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MakeDirs(FLAGS.data_dir)

  if not tf.gfile.Exists(FLAGS.checkpoint_dir):
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main(_):
  print_flags()

  initialize_folders()

  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
  parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
  parser.add_argument('--print_freq', type=int, default=PRINT_FREQ_DEFAULT,
                        help='Frequency of evaluation on the train set')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--checkpoint_freq', type=int, default=CHECKPOINT_FREQ_DEFAULT,
                        help='Frequency with which the model state is saved.')
  parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default=LOG_DIR_DEFAULT,
                        help='Summaries log directory')
  parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR_DEFAULT,
                        help='Checkpoint directory')
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run()
