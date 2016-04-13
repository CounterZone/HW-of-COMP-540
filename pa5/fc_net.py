import numpy as np

from layers import *
from layer_utils import *
import scipy.sparse

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of d, a hidden dimension of h, and perform classification over C classes.
  
  The architecture should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'theta1' and 'theta1_0' and second     #
    # layer weights and biases using the keys 'theta2' and 'theta2_0'.         #
    ############################################################################
    # 4 lines of code expected

    self.params['theta1']=np.random.normal(0,weight_scale,[input_dim,hidden_dim])
    self.params['theta1_0']=np.zeros([1,hidden_dim])
    self.params['theta2']=np.random.normal(0,weight_scale,[hidden_dim,num_classes])
    self.params['theta2_0']=np.zeros([1,num_classes])
    

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (m, d_1, ..., d_k)
    - y: Array of labels, of shape (m,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - output: Array of shape (m, C) giving classification scores, where
      output[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    output = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # output for X and storing them in the output variable.                    #
    ############################################################################
    # Hint: unpack the weight parameters from self.params
    # 3 lines of code expected
    theta1,theta1_0,theta2,theta2_0=self.params['theta1'],self.params['theta1_0'],self.params['theta2'],self.params['theta2_0']
    out1,c1=affine_relu_forward(X,theta1,theta1_0)
    output,c2=affine_forward(out1,theta2,theta2_0)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return output
    if y is None:
      return output
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # 8 lines of code expected
    ypre=np.exp(output)/(np.exp(output).sum(1).reshape(output.shape[0],1).dot(np.ones([1,output.shape[1]])))
    my=scipy.sparse.csr_matrix((np.ones(y.shape),y,np.arange(y.shape[0]+1)),shape=ypre.shape).todense()
    loss=(-1.0*np.sum(np.multiply(my,np.log(ypre)))/len(y)+(self.reg/2)*(np.sum(np.array(theta1)**2)+np.sum(np.array(theta2)**2)))
    grad2=(-1.0*((my-ypre)))/len(y)
    dx2,grads['theta2'],grads["theta2_0"]=affine_backward(grad2,c2)
    dx1,grads['theta1'],grads["theta1_0"]=affine_relu_backward(dx2, c1)
    grads['theta1']+=self.reg*theta1
    grads['theta2']+=self.reg*theta2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. 
  For a network with L layers,  the architecture will be
  
  {affine - [batch norm] - relu } x (L - 1) - affine - softmax
  
  and the {...} block is  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               reg=0.0,
               weight_scale=1e-2, dtype=np.float32):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.

    """

    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in theta1 and theta1_0; for the second layer use theta2 and theta2_0,    #
    # etc. Weights should be initialized from a normal distribution with       #
    # standard deviation equal to weight_scale and biases should be            #
    # initialized to zero.                                                     #
    ############################################################################
    # 7 lines of code expected
    self.params['theta1']=np.random.normal(0,weight_scale,[input_dim,hidden_dims[0]])
    self.params['theta1_0']=np.zeros([1,hidden_dims[0]])
    for i in range(2,self.num_layers):
	self.params['theta%d'%i]=np.random.normal(0,weight_scale,[hidden_dims[i-2],hidden_dims[i-1]])
	self.params["theta%d_0"%i]=np.zeros([1,hidden_dims[i-1]])
    self.params['theta%d'%self.num_layers]=np.random.normal(0,weight_scale,[hidden_dims[-1],num_classes])
    self.params["theta%d_0"%self.num_layers]=np.zeros([1,num_classes])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    output = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the output for X and storing them in the output variable.                #
    #                                                                          #
    ############################################################################
    # 6 lines of code expected.
    out=[]
    out.append(affine_relu_forward(X,self.params['theta1'],self.params['theta1_0']))
    for i in range(2,self.num_layers):
	out.append(affine_relu_forward(out[-1][0],self.params['theta%d'%i],self.params['theta%d_0'%i]))
    out.append(affine_forward(out[-1][0],self.params['theta%d'%self.num_layers],self.params['theta%d_0'%self.num_layers]))
    output=out[-1][0]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return output

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # 10-12 lines of code expected
    ypre=np.exp(out[-1][0])/(np.exp(out[-1][0]).sum(1).reshape(out[-1][0].shape[0],1).dot(np.ones([1,out[-1][0].shape[1]])))
    my=scipy.sparse.csr_matrix((np.ones(y.shape),y,np.arange(y.shape[0]+1)),shape=ypre.shape).todense()
    loss=-1.0*np.sum(np.multiply(my,np.log(ypre)))/len(y)+(self.reg/2)*np.sum(np.array(self.params['theta%d'%(self.num_layers)])**2)
    grad=(-1.0*((my-ypre)))/len(y)
    dx,grads['theta%d'%self.num_layers],grads["theta%d_0"%self.num_layers]=affine_backward(grad,out[-1][1])
    grads['theta%d'%self.num_layers]+=self.reg*(self.params['theta%d'%(self.num_layers)])
    for i in range(2,self.num_layers+1):
	dx,grads['theta%d'%(self.num_layers-i+1)],grads['theta%d_0'%(self.num_layers-i+1)]=affine_relu_backward(dx, out[-i][1])
	loss+=(self.reg/2)*np.sum(np.array(self.params['theta%d'%(self.num_layers-i+1)])**2)
        grads['theta%d'%(self.num_layers-i+1)]+=self.reg*(self.params['theta%d'%(self.num_layers-i+1)])


    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
