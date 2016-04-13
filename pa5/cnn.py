import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *
import scipy.sparse

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (m, C, H, W)
  consisting of m images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys      #
    # 'theta1' and 'theta1_0'; use keys 'theta2' and 'theta2_0' for the        #
    # weights and biases of the hidden affine layer, and keys 'theta3' and     #
    # 'theta3_0' for the weights and biases of the output affine layer.        #
    ############################################################################
    # about 15 lines of code
    self.params["theta1"]=np.random.normal(0,weight_scale,[num_filters,input_dim[0],filter_size,filter_size])
    self.params["theta1_0"]=np.zeros([num_filters])
    self.params["theta2"]=np.random.normal(0,weight_scale,[num_filters*input_dim[1]*input_dim[2]/4,hidden_dim])
    self.params["theta2_0"]=np.zeros([1,hidden_dim])
    self.params["theta3"]=np.random.normal(0,weight_scale,[hidden_dim,num_classes])
    self.params["theta3_0"]=np.zeros([1,num_classes])

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    theta1, theta1_0 = self.params['theta1'], self.params['theta1_0']
    theta2, theta2_0 = self.params['theta2'], self.params['theta2_0']
    theta3, theta3_0 = self.params['theta3'], self.params['theta3_0']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = theta1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # about 3 lines of code (use the helper functions in layer_utils.py)
    
    o1,c1=conv_relu_pool_forward(X, theta1, theta1_0, conv_param, pool_param)

    o2,c2=affine_relu_forward(o1, theta2, theta2_0)

    scores,c3=affine_forward(o2, theta3, theta3_0)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # about 12 lines of code.
    output=scores
    ypre=np.exp(output)/(np.exp(output).sum(1).reshape(output.shape[0],1).dot(np.ones([1,output.shape[1]])))
    my=scipy.sparse.csr_matrix((np.ones(y.shape),y,np.arange(y.shape[0]+1)),shape=ypre.shape).todense()
    loss=(-1.0*np.sum(np.multiply(my,np.log(ypre)))/len(y)+(self.reg/2)*(np.sum(np.array(theta1)**2)+np.sum(np.array(theta2)**2)+np.sum(np.array(theta3)**2)))
    grad3=(-1.0*((my-ypre)))/len(y)
    dx3,grads['theta3'],grads["theta3_0"]=affine_backward(grad3,c3)
    dx2,grads['theta2'],grads["theta2_0"]=affine_relu_backward(dx3, c2)
    dx1,grads['theta1'],grads["theta1_0"]=conv_relu_pool_backward(dx2, c1)
    grads['theta1']+=self.reg*theta1
    grads['theta2']+=self.reg*theta2
    grads['theta3']+=self.reg*theta3
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  

