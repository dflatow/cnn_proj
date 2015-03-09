# As usual, a bit of setup

import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifier_trainer import ClassifierTrainer
from cs231n.gradient_check import eval_numerical_gradient
from cs231n.classifiers.convnet_final import *
import cPickle as pickle
import os
from cs231n.data_utils import get_CIFAR10_data

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-18, np.abs(x) + np.abs(y))))


def permute_lables(p, labels_orig, classes=9): 
    labels = labels_orig.copy()
    n_labels = len(labels)
    n_to_permute = int(np.floor(p * n_labels))
    inds_to_permute = np.random.choice(n_labels, n_to_permute, replace=False)
    new_labels = np.random.choice(classes, n_to_permute, replace=True)
    labels[inds_to_permute] = new_labels
    return labels



if __name__ == "__main__":

  print "START"  
  X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
  print 'Train data shape: ', X_train.shape
  print 'Train labels shape: ', y_train.shape
  print 'Validation data shape: ', X_val.shape
  print 'Validation labels shape: ', y_val.shape
  print 'Test data shape: ', X_test.shape
  print 'Test labels shape: ', y_test.shape

  
  p = 0.00
  y_train_permuted = permute_lables(p, y_train)
  model = init_three_layer_convnet(filter_size=5, weight_scale=5e-3, num_filters=32)
  trainer = ClassifierTrainer()
  best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
          X_train, y_train_permuted, X_val, y_val, model, three_layer_convnet, update='momentum',
          reg=0.00008, momentum=0.9, learning_rate=0.0014, batch_size=300, num_epochs=30,
          verbose=True)

  fname = 'test_308_2.p'
  with open(fname, 'w+') as f:
      pickle.dump(model, f)

  print "DONE"