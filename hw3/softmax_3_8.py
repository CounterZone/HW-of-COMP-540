import numpy as np
from random import shuffle
import scipy.sparse

class SoftmaxClassifier:

  def __init__(self):
    self.theta = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, tol=1e-5, verbose=False):
    """
    Train the classifier using mini-batch stochastic gradient descent.

    Inputs:
    - X: m x d array of training data. Each training point is a d-dimensional
         row.
    - y: 1-dimensional array of length m with labels 0...K-1, for K classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) maximum number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - tol: (float) stop before num_iters if successive iteration loss is less than tol (tolerance)
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train,dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    self.theta = np.random.randn(dim,num_classes) * 0.001

    # Run stochastic gradient descent to optimize theta
    win_len = 5
    loss_history = [np.inf]*win_len
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      # Hint: 3 lines of code expected
      
      index=np.random.choice(range(0,len(y)),size=batch_size)
      X_batch=X[index,:]
      y_batch=y[index]


      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      
      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      # Hint: 1 line of code expected
      last_loss = np.mean(loss_history[-win_len:])
      loss_history.append(loss)
      curr_loss = np.mean(loss_history[-win_len:])
      if abs(last_loss - curr_loss) < tol:
        self.theta-=grad*learning_rate
        return it, loss_history

      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

    return num_iters, loss_history
    
  def train_fmin(self, X, y, reg=1e-5, num_iters=100):
    """
    Train the classifier using mini-batch stochastic gradient descent.

    Inputs:
    - X: m x d array of training data. Each training point is a d-dimensional
         row.
    - y: 1-dimensional array of length m with labels 0...K-1, for K classes.
    - reg: (float) regularization strength.
    - num_iters: (integer) maximum number of steps to take when optimizing

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train,dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    self.theta = np.random.randn(dim,num_classes) * 0.001
    
    self.theta = scipy.optimize.fmin_bfgs(softmax_fmin_loss, self.theta, fprime = softmax_fmin_grad_loss, args=(X,y,reg),maxiter=num_iters)

    # evaluate loss and gradient
    loss, grad = self.loss(X, y, reg)

    return loss

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: m x d array of training data. Each row is a d-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length m, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[0])
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    # Hint: 1 line of code expected

    y_pred=np.argmax(X.dot(self.theta),1)

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: m x d array of data; each row is a data point.
    - y_batch: 1-dimensional array of length m with labels 0...K-1, for K classes.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.theta; an array of the same shape as theta
    """

    return softmax_loss_vectorized(self.theta, X_batch, y_batch, reg)

  
def softmax_loss_naive(theta, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in J and the gradient in grad. If you are not              #
  # careful here, it is easy to run into numeric instability. Don't forget    #
  # the regularization term!                                                  #
  #############################################################################
  # Hint: about 5-10 lines of code expected

  for i in range(0,m):
    p=np.zeros(max(y)+1)
    for j in range(0,max(y)+1):
	po=0
	for jj in range(0,max(y)+1):
		po=po+np.exp(theta[:,jj].dot(X[i,:])-theta[:,j].dot(X[i,:]))
	p[j]=1/po
    	grad[:,j]-=X[i,:]*(float(y[i]==j)-p[j])/m
    J=J+np.log(p[y[i]])
  J=-J/m+(reg/2/m)*np.sum(theta**2)
  grad=grad+ theta*reg/m

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad

def convert_y_to_matrix(y):
  """
  convert an array of m elements with values in {0,...,K-1} to a boolean matrix
  of size m x K where there is a 1 for the value of y in that row.

  """
  y = np.array(y)
  data = np.ones(len(y))
  indptr = np.arange(len(y)+1)
  mat = scipy.sparse.csr_matrix((data,y,indptr))
  return mat.todense()

  
def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in J and the gradient in grad. If you are not careful      #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization term!                                                      #
  #############################################################################
  # Hint: 4-6 lines of code expected
  xt=X.dot(theta)
  Pt=np.exp(xt-np.max(xt,1).reshape([m,1]).dot(np.ones([1,theta.shape[1]])))
  P=Pt/Pt.sum(1).reshape([m,1]).dot(np.ones([1,theta.shape[1]]))
  J=-1.0/m*np.sum(np.multiply(np.log(P),convert_y_to_matrix(y)))+(reg/2/m)*np.sum(theta**2)
  grad=-1.0/m*X.T.dot((convert_y_to_matrix(y)-P))+ theta*reg/m

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad


def softmax_fmin_loss(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss to zero.

  J = 0.0
  m, dim = X.shape

  xt=X.dot(theta)
  Pt=np.exp(xt-np.max(xt,1).reshape([m,1]).dot(np.ones([1,theta.shape[1]])))
  P=Pt/Pt.sum(1).reshape([m,1]).dot(np.ones([1,theta.shape[1]]))
  J=-1.0/m*np.sum(np.multiply(np.log(P),convert_y_to_matrix(y)))+(reg/2/m)*np.sum(theta**2)

  return J
  
def softmax_fmin_grad_loss(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the gradient to zero.

  grad = np.zeros_like(theta)
  m, dim = X.shape


  xt=X.dot(theta)
  Pt=np.exp(xt-np.max(xt,1).reshape([m,1]).dot(np.ones([1,theta.shape[1]])))
  P=Pt/Pt.sum(1).reshape([m,1]).dot(np.ones([1,theta.shape[1]]))
  grad=-1.0/m*X.T.dot((convert_y_to_matrix(y)-P))+ theta*reg/m

  return grad