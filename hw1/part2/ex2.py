########################################################################
# comp 540, Spring 2016                                                #
# Supervised Machine Learning                                          #
# Devika Subramanian, Rice University                                  #
#                                                                      #
# Homework 1: Part 2: Regularized linear regression                    #
#                     with multiple variables                          #
########################################################################

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on 
#  linear regression with regularization. 
#  You will need to complete functions in
#  reg_linear_regressor.py, ex2.py and utils.py
#  There is no need to modify this script.

import numpy as np
import matplotlib.pyplot as plt
import utils
import plot_utils
from reg_linear_regressor_multi import RegularizedLinearReg_SquaredLoss

########################################################################
## =========== Part 1: Loading and Visualizing Data ===================#
########################################################################
#  We start the exercise by first loading and visualizing the dataset. #
#  The following code will load the dataset into your environment and  #
#  plot the data.                                                      #
########################################################################


# Load Training Data

print 'Loading and Visualizing Data ...'

X, y, Xtest, ytest, Xval, yval = utils.load_mat('ex2data1.mat')

# Plot training data

plot_utils.plot_data(X,y,'Change in water level (x)','Water flowing out of the dam (y)')
plt.savefig('fig6.pdf')

########################################################################
## =========== Part 2: Regularized Linear Regression ==================#
########################################################################
#  You should now implement the loss function and gradient of the
# loss function for regularized linear regression in reg_linear_regression_multi.py

# append a column of ones to matrix X

XX = np.vstack([np.ones((X.shape[0],)),X]).T

#  Train linear regression with lambda = 0

reglinear_reg1 = RegularizedLinearReg_SquaredLoss()
theta_opt0 = reglinear_reg1.train(XX,y,reg=0.0,num_iters=1000)
print 'Theta at lambda = 0 is ', theta_opt0

# plot fit over data and save it in fig7.pdf

plt.plot(X,np.dot(XX,theta_opt0),'g-',linewidth=3)
plt.savefig('fig7.pdf')

#######################################################################
# =========== Part 3: Learning Curve for Linear Regression ===========#
#######################################################################

reg = 1.0
XXval = np.vstack([np.ones((Xval.shape[0],)),Xval]).T

# implement the learning_curve function in utils.py
# this script will run your function and save the curves in fig8.pdf

error_train, error_val = utils.learning_curve(XX,y,XXval,yval,reg)
plot_utils.plot_learning_curve(error_train, error_val,reg)
plt.savefig('fig8.pdf')

#######################################################################
## =========== Part 4: Feature Mapping for Polynomial Regression =====#
#######################################################################

from utils import feature_normalize
import sklearn
from sklearn.preprocessing import PolynomialFeatures

# Map X onto polynomial features and normalize
# We will consider a 6th order polynomial fit for the data

p = 6
poly = sklearn.preprocessing.PolynomialFeatures(degree=p,include_bias=False)
X_poly = poly.fit_transform(np.reshape(X,(len(X),1)))
X_poly, mu, sigma = utils.feature_normalize(X_poly)

# add a column of ones to X_poly

XX_poly = np.vstack([np.ones((X_poly.shape[0],)),X_poly.T]).T

# map Xtest and Xval into the same polynomial features

X_poly_test = poly.fit_transform(np.reshape(Xtest,(len(Xtest),1)))
X_poly_val = poly.fit_transform(np.reshape(Xval,(len(Xval),1)))

# normalize these two sets with the same mu and sigma

X_poly_test = (X_poly_test - mu) / sigma
X_poly_val = (X_poly_val - mu) / sigma

# add a column of ones to both X_poly_test and X_poly_val
XX_poly_test = np.vstack([np.ones((X_poly_test.shape[0],)),X_poly_test.T]).T
XX_poly_val = np.vstack([np.ones((X_poly_val.shape[0],)),X_poly_val.T]).T

#######################################################################
## =========== Part 5: Learning Curve for Polynomial Regression ======#
#######################################################################

reg = 0.0
reglinear_reg2 = RegularizedLinearReg_SquaredLoss()
theta_opt1 = reglinear_reg1.train(XX_poly,y,reg=reg,num_iters=10000)
print 'Theta at lambda = 0 is ', theta_opt1


# plot data and training fit for the 6th order polynomial and save it in fig9.pdf

plot_utils.plot_fit(X,y,np.min(X),np.max(X),mu,sigma,theta_opt1,p,'Change in water level (x)','Water flowing out of dam (y)','Polynomial Regression fit with lambda = 0 and polynomial features of degree = ' + str(p))

plt.savefig('fig9.pdf')

# plot learning curve for data (6th order polynomail basis function) and save
# it in fig10.pdf

error_train,error_val = utils.learning_curve(XX_poly,y,XX_poly_val,yval,reg)
plot_utils.plot_learning_curve(error_train,error_val,reg)
plt.savefig('fig10.pdf')


#######################################################################
## =========== Part 6: Averaged learning curve         ===============#
#######################################################################

# now implement the averaged learning curve function in utils.py
# The script runs your function, plots the curves and saves it in fig11.pdf

reg = 1.0
error_train,error_val = utils.averaged_learning_curve(XX_poly,y,XX_poly_val,yval,reg)
plot_utils.plot_learning_curve(error_train,error_val,reg)
plt.savefig('fig11.pdf')


#######################################################################
## =========== Part 6: Selecting Lambda                ===============#
#######################################################################

# now implement the validation_curve function in utils.py
# this function helps in determining the best lambda using a
# a validation set
# The script will now run your function and plot the figure in fig12.pdf

reg_vec, error_train, error_val = utils.validation_curve(XX_poly,y,XX_poly_val,yval)
plot_utils.plot_lambda_selection(reg_vec,error_train,error_val)
plt.savefig('fig12.pdf')
