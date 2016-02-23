import random
import numpy as np
import matplotlib.pyplot as plt
import utils
from softmax_3_8 import softmax_loss_naive, softmax_loss_vectorized
from softmax_3_8 import SoftmaxClassifier
import time




# Get the CIFAR-10 data broken up into train, validation and test sets

X_train, y_train, X_val, y_val, X_test, y_test = utils.get_CIFAR10_data()

# First implement the naive softmax loss function with nested loops.
# Open the file softmax.py and implement the
# softmax_loss_naive function.

# Generate a random softmax theta matrix and use it to compute the loss.

theta = np.random.randn(3073,10) * 0.0001

results = {}
best_val = -1
best_bs = -1
best_iter = -1
best_lr = -1
best_reg = -1
best_softmax = None
'''
batch_sizes = [100, 200]
learning_rates = [1e-7, 5e-7]
regularization_strengths = [5e4, 1e5]
'''
regularization_strengths = [5e4, 1e5, 5e5, 1e8]


################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# Save the best trained softmax classifer in best_softmax.                     #
# Hint: about 10 lines of code expected
################################################################################
ns=SoftmaxClassifier()
max_iters=8000

for rs in regularization_strengths:
    ns.train_fmin(X_train,y_train,rs,max_iters)
    ta=np.mean(y_train == ns.predict(X_train))
    va=np.mean(y_val == ns.predict(X_val))
    results[rs]=(ta,va)
    if va>best_val:
	best_val=va
	best_reg = rs
	best_softmax=ns
	print '\t---- FINISHED fmin with reg: %e------' %rs
################################################################################
#                              END OF YOUR CODE                                #
################################################################################
    
# Print out results.
for rs in sorted(results):
    train_accuracy, val_accuracy = results[(rs)]
    print 'reg %e train accuracy: %f val accuracy: %f' % (
                rs, train_accuracy, val_accuracy)
    
print 'best validation accuracy achieved during cross-validation: %f' % best_val
print '\t with reg %e' %best_reg



# Evaluate the best softmax classifier on test set

if best_softmax:
  y_test_pred = best_softmax.predict(X_test)
  test_accuracy = np.mean(y_test == y_test_pred)
  print 'softmax on raw pixels final test set accuracy: %f' % (test_accuracy, )

  # Visualize the learned weights for each class

  theta = best_softmax.theta[1:,:].T # strip out the bias term
  theta = theta.reshape(10, 32, 32, 3)

  theta_min, theta_max = np.min(theta), np.max(theta)

  classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  for i in xrange(10):
    plt.subplot(2, 5, i + 1)
  
    # Rescale the weights to be between 0 and 255
    thetaimg = 255.0 * (theta[i].squeeze() - theta_min) / (theta_max - theta_min)
    plt.imshow(thetaimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])


  plt.savefig('cifar_theta_3_8_fmin.pdf')
  plt.close()
