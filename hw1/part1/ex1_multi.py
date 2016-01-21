########################################################################
# comp 540                                                             #
# Statistical Machine Learning                                         #
# Devika Subramanian, Rice University                                  #
#                                                                      #
# Homework 1: Part A2: Linear Regression with the Boston Housing data  #
#             Linear regression with multiple variables                #
########################################################################

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on 
#  linear regression with many variables. 
#  You will need to complete functions 
#  in linear_regressor_multi.py and utils.py.
#  The only changes to make in this file are marked with TODO:

########################################################################
##================ Part 0: Reading data              ==================#
########################################################################

from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_utils, utils
from linear_regressor_multi import LinearRegressor_Multi, LinearReg_SquaredLoss

print 'Reading data ...'
bdata = load_boston()
df = pd.DataFrame(data = bdata.data, columns = bdata.feature_names)


########################################################################
# ======= Part 2: Linear regression with multiple variables ===========#
########################################################################

X = df.values
y = bdata.target

# need to scale the features (use zero mean scaling)

X_norm,mu,sigma = utils.feature_normalize(X)

# add intercept term to X_norm

XX = np.vstack([np.ones((X.shape[0],)),X_norm.T]).T

print 'Running gradient descent ..'

# set up model and train 

linear_reg3 = LinearReg_SquaredLoss()
J_history3 = linear_reg3.train(XX,y,learning_rate=0.01,num_iters=5000,verbose=False)

# Plot the convergence graph and save it in fig5.pdf

plot_utils.plot_data(range(len(J_history3)),J_history3,'Number of iterations','Cost J')
plt.savefig('fig5.pdf')

# Display the computed theta

print 'Theta computed by gradient descent: ', linear_reg3.theta


########################################################################
# ======= Part 3: Predict on unseen data with model ======= ===========#
########################################################################

########################################################################
# TODO:                                                                #
# Predict values for the average home                                  #
# remember to multiply prediction by 10000 using linear_reg3           #
#   One line of code expected; replace pred_cost = 0 line              # 
########################################################################

pred_cost = 0
print 'For average home in Boston suburbs, we predict a median home value of', pred_cost


########################################################################
# ============= Part 4: Solving with normal equations =================#
########################################################################

X = df.values
y = bdata.target
XX1 = np.vstack([np.ones((X.shape[0],)),X.T]).T

linear_reg4 = LinearReg_SquaredLoss()


theta_n = linear_reg4.normal_equation(XX1,y)

print 'Theta computed by direct solution is: ', theta_n

########################################################################
# TODO:                                                                #
# Predict values for the average home using theta_n                    #
# remember to multiply prediction by 10000                             #
#   One line of code expected; replace pred_cost = 0 line              # 
########################################################################

pred_cost = 0
print 'For average home in Boston suburbs, we predict a median home value of', pred_cost

########################################################################
##========== Part 5:   Exploring convergence rates=====================#
########################################################################

# change the learning_rate and num_iters in the call below to find the 
# best learning rate for this data set.

learning_rates = [0.01, 0.03, 0.1, 0.3]

########################################################################
# TODO:                                                                #
# Produce convergence plots for gradient descent with the rates above  #
# using data (XX,y). Include them in your writeup.                     #
#   4-5 lines of code expected                                         #
########################################################################

