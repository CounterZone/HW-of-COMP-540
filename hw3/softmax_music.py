import random
import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD
import music_utils
=======
>>>>>>> 89dd6a53aa0ff700b713b57c5d8d001424557b1d
import utils
from softmax import softmax_loss_naive, softmax_loss_vectorized
from softmax import SoftmaxClassifier
import time
<<<<<<< HEAD
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
# TODO: Get the music dataset (CEFS representation) [use code from Hw2]

# some global constants

MUSIC_DIR = "music/"
genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

# select the CEPS or FFT representation

X,y = music_utils.read_ceps(genres,MUSIC_DIR)
=======

# TODO: Get the music dataset (CEFS representation) [use code from Hw2]
>>>>>>> 89dd6a53aa0ff700b713b57c5d8d001424557b1d


# TODO: Split into train, validation and test sets 

<<<<<<< HEAD
X_tr, X_test, y_tr, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
X_train,X_val,y_train,y_val= cross_validation.train_test_split(X_tr, y_tr, test_size=0.1)

=======
>>>>>>> 89dd6a53aa0ff700b713b57c5d8d001424557b1d

# TODO: Use the validation set to tune hyperparameters for softmax classifier
# choose learning rate and regularization strength (use the code from softmax_hw.py)


<<<<<<< HEAD
results = {}
best_val = -1
best_softmax = None
learning_rates = [1e-4,1e-3,1e-2,1e-1]
regularization_strengths = [0.01,0.05,0.1,0.5,1]
for lr in learning_rates:
	for rs in regularization_strengths:
		print("calculating: lr=%e,reg=%e"%(lr,rs))
		ns=SoftmaxClassifier()
		ns.train(X_train,y_train,lr,rs,batch_size=400,num_iters=2000)
		ta=np.mean(y_train == ns.predict(X_train))
		va=np.mean(y_val == ns.predict(X_val))
		results[lr,rs]=(ta,va)
		if va>best_val:
			best_val=va
			best_softmax=ns



# TODO: Evaluate best softmax classifier on set aside test set (use the code from softmax_hw.py)
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy)
    
print 'best validation accuracy achieved during cross-validation: %f' % best_val

# Evaluate the best softmax classifier on test set

if best_softmax:
  y_test_pred = best_softmax.predict(X_test)
  test_accuracy = np.mean(y_test == y_test_pred)
  print 'softmax on raw pixels final test set accuracy: %f' % (test_accuracy, )
  cm=confusion_matrix(y_test, y_test_pred)
  print(cm)
# TODO: Compare performance against OVA classifier of Homework 2 with the same
# train, validation and test sets (use sklearn's classifier evaluation metrics)

overall_accuracy = cm.trace()/float(cm.sum())

print "--- Overall accuracy with Mel Cepstral representation", overall_accuracy


genre_accuracy = np.multiply(cm.diagonal() , 1.0/cm.sum(1))


for i in range(0,10):
    print "--- Genre = ", genres[i], " accuracy with Mel Cepstral representation", genre_accuracy[i]





=======
# TODO: Evaluate best softmax classifier on set aside test set (use the code from softmax_hw.py)


# TODO: Compare performance against OVA classifier of Homework 2 with the same
# train, validation and test sets (use sklearn's classifier evaluation metrics)
>>>>>>> 89dd6a53aa0ff700b713b57c5d8d001424557b1d
