import numpy as np

def svm_loss_twoclass(theta, X, y, C):
  """
  SVM hinge loss function for two class problem

  Inputs:
  - theta: A numpy vector of size d containing coefficients.
  - X: A numpy array of shape mxd 
  - y: A numpy array of shape (m,) containing training labels; +1, -1
  - C: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to theta; an array of same shape as theta
"""

  m, d = X.shape
  grad = np.zeros(theta.shape)
  J = 0

  ######################################################################
  # TODO                                                               #
  # Compute loss J and gradient of J with respect to theta             #
  # 2-3 lines of code expected                                         #
  ######################################################################
<<<<<<< HEAD


=======
  p = np.dot(X,theta)
  J =  (1.0/2/m)*np.linalg.norm(theta,2)**2 + C* (((1-y*p ) >0) * (1.0-y*p)/m ).sum()
  grad = (1.0/m)*theta - (C/m)*np.dot(X.T , y* ((1-y*p) > 0))
>>>>>>> 90033ce... 3.3 completed3.3 completed3.3 completed
  ######################################################################
  # end of your code                                                   #
  ######################################################################
  return J, grad

