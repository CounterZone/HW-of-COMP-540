import numpy as np
import fast_layers

def affine_forward(x, theta, theta_0):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (m, d_1, ..., d_k) and contains a minibatch of m
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension d = d_1 * ... * d_k, and
  then transform it to an output vector of dimension h.

  Inputs:
  - x: A numpy array containing input data, of shape (m, d_1, ..., d_k)
  - theta: A numpy array of weights, of shape (d, h)
  - theta_0: A numpy array of biases, of shape (h,)
  
  Returns a tuple of:
  - out: output, of shape (m, h)
  - cache: (x, theta, theta_0)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  # 2 lines of code expected

  X=x.reshape(x.shape[0],theta.shape[0])
  out=X.dot(theta)+np.ones([x.shape[0],1]).dot(theta_0.reshape(1,theta_0.size))
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, theta, theta_0)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (m, h)
  - cache: Tuple of:
    - x: Input data, of shape (m, d_1, ... d_k)
    - theta: Weights, of shape (d, h)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (m, d1, ..., d_k)
  - dtheta: Gradient with respect to theta, of shape (d, h)
  - dtheta_0: Gradient with respect to b, of shape (h,)
  """
  x, theta, theta_0 = cache
  dx, dtheta, dtheta_0 = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  # Hint: do not forget to reshape x into (m,d) form
  # 4-5 lines of code expected
  X=x.reshape(x.shape[0],theta.shape[0])
  dx=np.array(dout.dot(theta.T)).reshape(x.shape)
  dtheta=X.T.dot(dout)
  dtheta_0=np.ones([1,x.shape[0]]).dot(dout)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dtheta, dtheta_0


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  # 1-2 lines of code expected.
  out=np.multiply(x,(x>0))

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  # 1-2 lines of code expected. Hint: use np.where

  dx=np.multiply(dout,(x>0))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx




def conv_forward_naive(x, theta, theta0, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of m data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (m, C, H, W)
  - theta: Filter weights of shape (K, C, HH, WW)
  - theta0: Biases, of shape (K,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (m, K, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, theta, theta0, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  # 14 lines of code expected
  m,C,H,W=x.shape
  K,C,HH,WW=theta.shape
  Hp= 1 + (H + 2 * conv_param['pad'] - HH) / conv_param['stride']
  Wp= 1 + (W + 2 * conv_param['pad'] - WW) / conv_param['stride']
  t1=np.zeros([m,C,H,conv_param['pad']])
  XX=np.concatenate((t1,x,t1),axis=3)
  t1=np.zeros([m,C,conv_param['pad'],W+2*conv_param['pad']])
  XX=np.concatenate((t1,XX,t1),axis=2)
  out=np.zeros([m,K,Hp,Wp])
  for i in range(Hp):
	for j in range(Wp):
			out[:,:,i,j]=XX[:,:,i*conv_param['stride']:i*conv_param['stride']+HH,j*conv_param['stride']:j*conv_param['stride']+WW].reshape([m,C*HH*WW]).dot(np.matrix(theta.reshape([K,C*HH*WW])).T)+np.ones([m,1])*theta0.reshape([1,K])
		

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, theta, theta0, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive


- x: Input data of shape (m, C, H, W)
  - theta: Filter weights of shape (K, C, HH, WW)
  - theta0: Biases, of shape (K,)


  Returns a tuple of:
  - dx: Gradient with respect to x
  - dtheta: Gradient with respect to theta
  - dtheta0: Gradient with respect to theta0
  """
  dx, dtheta, dtheta0 = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  # 20-22 lines of code expected

  x, theta, theta0, conv_param=cache 
  m,K,Hp,Wp=dout.shape
  m,C,H,W=x.shape
  K,C,HH,WW=theta.shape
  t1=np.zeros([m,C,H,conv_param['pad']])
  XX=np.concatenate((t1,x,t1),axis=3)
  t1=np.zeros([m,C,conv_param['pad'],W+2*conv_param['pad']])
  XX=np.concatenate((t1,XX,t1),axis=2)
  dX=np.zeros(XX.shape)
  dtheta=np.zeros(theta.shape)
  dtheta0=dout.sum(0).sum(1).sum(1)
  for i in range(Hp):
	for j in range(Wp):
		dX[:,:,i*conv_param['stride']:i*conv_param['stride']+HH,j*conv_param['stride']:j*conv_param['stride']+WW]+=(dout[:,:,i,j].dot(theta.reshape(K,C*HH*WW))).reshape(m,C,HH,WW)
		
		dtheta+=np.array(np.matrix(dout[:,:,i,j]).T.dot(XX[:,:,i*conv_param['stride']:i*conv_param['stride']+HH,j*conv_param['stride']:j*conv_param['stride']+WW].reshape([m,C*HH*WW]))).reshape([K,C,HH,WW])
  dx=dX[:,:,conv_param['pad']:-conv_param['pad'],conv_param['pad']:-conv_param['pad']]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dtheta, dtheta0


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (m, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  # 12-13 lines of code expected
  
  m,C,H,W=x.shape
  HH,WW,S=pool_param["pool_height"],pool_param["pool_width"],pool_param["stride"]
  Hp= 1 + (H - HH) / S
  Wp= 1 + (W - WW) / S
  out=np.zeros([m,C,Hp,Wp])
  for i in range(Hp):
	for j in range(Wp):
			out[:,:,i,j]=np.max(np.max(x[:,:,i*S:i*S+HH,j*S:j*S+WW],2),2)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  # 15 lines of code expected
  x,pool_param=cache
  HH,WW,S=pool_param["pool_height"],pool_param["pool_width"],pool_param["stride"]
  m,C,H,W=x.shape
  m,C,Hp,Wp=dout.shape
  dx=np.zeros(x.shape)
  for i in range(Hp):
	for j in range(Wp):
		for c in range(C):
			for mm in range(m):
				ii=np.argmax(x[mm,c,i*S:i*S+HH,j*S:j*S+WW])%WW
				dx[mm,c,(np.argmax(x[mm,c,i*S:i*S+HH,j*S:j*S+WW])-ii)/HH+i*S,ii+j*S]+=dout[mm,c,i,j]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx



def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (m, C) where x[i, j] is the output for the jth class
    for the ith input.
  - y: Vector of labels, of shape (m,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  m = x.shape[0]
  correct_class_output = x[np.arange(m), y]
  margins = np.maximum(0, x - correct_class_output[:, np.newaxis] + 1.0)
  margins[np.arange(m), y] = 0
  loss = np.sum(margins) / m
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(m), y] -= num_pos
  dx /= m
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (m, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (m,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  m = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(m), y])) / m
  dx = probs.copy()
  dx[np.arange(m), y] -= 1
  dx /= m
  return loss, dx



