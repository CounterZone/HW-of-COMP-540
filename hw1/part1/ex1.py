########################################################################
# comp 540                                                             #
# Statistical Machine Learning                                         #
# Devika Subramanian, Rice University                                  #
#                                                                      #
# Homework 1: Part A: Linear Regression with the Boston Housing data   #
#             Single variable linear regression                        #
########################################################################

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear regression. You will need to complete the functions 
#  in linear_regressor.py and utils.py in the places indicated.
#  Modify this file at the places marked TODO:

########################################################################
##================ Part 0: Reading data and plotting ==================#
########################################################################

from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt

print 'Reading data ...'
bdata = load_boston()
df = pd.DataFrame(data = bdata.data, columns = bdata.feature_names)

#  X is the percentage of the population in a census tract that is of
#  lower economic status. X is a vector of length 506.
#  y is to the median home value in $10000's. y is a vector of length 506

X = df.LSTAT
y = bdata.target

# Scatter plot LSTAT vs median home value, saved in fig1.pdf

import numpy as np
import plot_utils
print 'Plotting data ...'
plot_utils.plot_data(X,y,'Percentage of population with lower economic status','Median home value in $10000s')
plt.savefig('fig1.pdf')

########################################################################
##============= Part 1: Training a univariate model ===================#
########################################################################

# Predict median home value from percentage of lower economic status in a census tract

# add the column of ones to X to represent the intercept term

XX = np.vstack([np.ones((X.shape[0],)),X]).T

from linear_regressor import LinearRegressor, LinearReg_SquaredLoss

# set up a linear regression model

linear_reg1 = LinearReg_SquaredLoss()

# run gradient descent

J_history1 = linear_reg1.train(XX,y,learning_rate=0.005,num_iters=10000,verbose=True)

# print the theta found

print 'Theta found by gradient_descent: ',linear_reg1.theta

# plot the linear fit and save it in fig2.pdf

plt.plot(X, np.dot(XX,linear_reg1.theta), 'g-',linewidth=3)
plt.savefig('fig2.pdf')

# Plot the convergence graph and save it in fig4.pdf

plot_utils.plot_data(range(len(J_history1)),J_history1,'Number of iterations','Cost J')
plt.savefig('fig4.pdf')


########################################################################
##============= Part 2: Predicting with the model   ===================#
########################################################################

# Predict values for lower status percentage of 5% and 50%
# remember to multiply prediction by 10000 because median value is in 10000s

###########################################################################
#   TODO:                                                                 #
#   Predicted median value of a home with LSTAT = 5%                      #
#   Hint: call the predict method with the appropriate x                  #
#         One line of code expected; replace line pred_cost = 0           #
###########################################################################

pred_cost = 0
print 'For lower status percentage = 5, we predict a median home value of', pred_cost

###########################################################################
#   TODO:                                                                 #
#   Predicted median value of a home with LSTAT = 50%                     #
#      One line of code expected, replace pred_cost = 0                   #
###########################################################################

pred_cost = 0
print 'For lower status percentage = 50, we predict a median home value of',pred_cost

########################################################################
# ============= Part 3: Visualizing J(theta_0, theta_1) ===============#
########################################################################

print 'Visualizing J(theta_0, theta_1) ...'

# Compute grid over which we will calculate J

theta0_vals = np.arange(-20,40, 0.1);
theta1_vals = np.arange(-4, 4, 0.1);
J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))

# Fill out J_vals and save plots in fig3a.pdf and fig3b.pdf

linear_reg2 = LinearReg_SquaredLoss()

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
      linear_reg2.theta = np.array([theta0_vals[i], theta1_vals[j]]).T 
      J_vals[i,j],_ = linear_reg2.loss(XX,y)
          
# Surface and contour plots

# Need to transpose J_vals before calling plot functions

J_vals = J_vals.T
tt1,tt2 = np.meshgrid(theta0_vals,theta1_vals)
plot_utils.make_surface_plot(tt1,tt2,J_vals,'$Theta_0$','$Theta_1$')
plt.savefig('fig3a.pdf')
plot_utils.make_contour_plot(tt1,tt2,J_vals,np.logspace(-10,40,200),'$Theta_0$','$Theta_1$',linear_reg1.theta)

plt.savefig('fig3b.pdf')

########################################################################
# ============= Part 4: Using sklearn's linear_model ==================#
########################################################################

# Check if the model you learned using gradient descent matches the one
# that sklearn's linear regression model learns on the same data.

from sklearn import linear_model
lr = linear_model.LinearRegression()
lr.fit(XX,y)

print "The coefficients computed by sklearn: ", lr.intercept_, " and ", lr.coef_[1]



