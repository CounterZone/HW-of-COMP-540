import numpy as np
import matplotlib.pyplot as plt

#############################################################################
#  Normalize features of data matrix X so that every column has zero        #
#  mean and unit variance                                                   #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     Output: mu: D x 1 (mean of X)                                         #
#          sigma: D x 1 (std dev of X)                                      #
#         X_norm: N x D (normalized X)                                      #
#############################################################################

def feature_normalize(X):

    ########################################################################
    # TODO: modify the three lines below to return the correct values

    mu = np.dot(np.ones([1,X.shape[0]]),X)/X.shape[0]
    sigma = (np.dot(np.ones([1,X.shape[0]]),(X-np.dot(np.ones([X.shape[0],1]),mu))**2)/X.shape[0])**0.5
    X_norm = (X-np.dot(np.ones([X.shape[0],1]),mu))/np.dot(np.ones([X.shape[0],1]),sigma)
  
    ########################################################################
    return X_norm, mu, sigma


