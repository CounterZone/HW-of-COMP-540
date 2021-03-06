import numpy as np
import linear_svm
import matplotlib.pyplot as plt
import utils
from sklearn import preprocessing, metrics
from linear_classifier import LinearSVM_twoclass

############################################################################
#  Part  0: Loading and Visualizing Data                                   #
#  We start the exercise by first loading and visualizing the dataset.     #
#  The following code will load the dataset into your environment and plot #
#  the data.                                                               #
############################################################################

# load ex6data1.mat

X,y = utils.load_mat('data/ex4data1.mat')

utils.plot_twoclass_data(X,y,'x1', 'x2',['neg','pos'])
plt.savefig('fig1.pdf')

<<<<<<< HEAD
=======

>>>>>>> 89dd6a53aa0ff700b713b57c5d8d001424557b1d
############################################################################
#  Part 1: Hinge loss function and gradient                                #
############################################################################

<<<<<<< HEAD
C = 1.0
=======
C = 1
>>>>>>> 89dd6a53aa0ff700b713b57c5d8d001424557b1d
theta = np.zeros((X.shape[1],))
J,grad = linear_svm.svm_loss_twoclass(theta,X,y,C)

print "J = ", J, " grad = ", grad

############################################################################
# Scale the data and set up the SVM training                               #
############################################################################

# scale the data

scaler = preprocessing.StandardScaler().fit(X)
scaleX = scaler.transform(X)

# add an intercept term and convert y values from [0,1] to [-1,1]

XX = np.array([(1,x1,x2) for (x1,x2) in scaleX])
yy = np.ones(y.shape)
yy[y == 0] = -1
yy[y == 0] = -1

############################################################################
#  Part  2: Training linear SVM                                            #
#  We train a linear SVM on the data set and the plot the learned          #
#  decision boundary                                                       #
############################################################################

############################################################################
# TODO                                                                     #
# You will change this line to vary C.                                     #
############################################################################

<<<<<<< HEAD
C = 100.0
=======
C = 1
>>>>>>> 89dd6a53aa0ff700b713b57c5d8d001424557b1d

############################################################################

svm = LinearSVM_twoclass()
svm.theta = np.zeros((XX.shape[1],))
svm.train(XX,yy,learning_rate=1e-4,C=C,num_iters=50000,verbose=False)

# classify the training data

y_pred = svm.predict(XX)

print "Accuracy on training data = ", metrics.accuracy_score(yy,y_pred)

# visualize the decision boundary

utils.plot_decision_boundary(scaleX,y,svm,'x1','x2',['neg','pos'])
plt.savefig('fig2.pdf')

############################################################################
#  Part  3: Training SVM with a kernel                                     #
#  We train an SVM with an RBF kernel on the data set and the plot the     #
#  learned decision boundary                                               #
############################################################################

# test your Gaussian kernel implementation

x1 = np.array([1,2,1])
x2 = np.array([0,4,-1])
<<<<<<< HEAD
sigma = 2.0
=======
sigma = 2
>>>>>>> 89dd6a53aa0ff700b713b57c5d8d001424557b1d

print "Guassian kernel value (should be around 0.324652) = ", utils.gaussian_kernel(x1,x2,sigma)

# load ex4data2.mat

X,y = utils.load_mat('data/ex4data2.mat')

# visualize the data

utils.plot_twoclass_data(X,y,'', '',['neg','pos'])
plt.savefig('fig3.pdf')

# convert X to kernel form with the kernel function

sigma = 0.02

# compute the kernel (slow!)

K = np.array([utils.gaussian_kernel(x1,x2,sigma) for x1 in X for x2 in X]).reshape(X.shape[0],X.shape[0])

# scale the kernelized data matrix

scaler = preprocessing.StandardScaler().fit(K)
scaleK = scaler.transform(K)

# add the intercept term

KK = np.vstack([np.ones((scaleK.shape[0],)),scaleK]).T

# transform y from [0,1] to [-1,1]

yy = np.ones(y.shape)
yy[y == 0] = -1

# set up the SVM and learn the parameters

svm = LinearSVM_twoclass()
svm.theta = np.zeros((KK.shape[1],))
<<<<<<< HEAD
C = 1.0
svm.train(KK,yy,learning_rate=1e-4,C=C,num_iters=20000,verbose=False)
=======
<<<<<<< HEAD
C = 1
svm.train(KK,yy,learning_rate=1e-4,C=C,num_iters=20000,verbose=True)
=======
C = 1.0
svm.train(KK,yy,learning_rate=1e-4,C=C,num_iters=20000,verbose=False)
>>>>>>> 90033ce... 3.3 completed3.3 completed3.3 completed
>>>>>>> 89dd6a53aa0ff700b713b57c5d8d001424557b1d

# visualize the boundary

utils.plot_decision_kernel_boundary(X,y,scaler,sigma,svm,'','',['neg','pos'])
plt.savefig("fig4.pdf")

<<<<<<< HEAD

=======
>>>>>>> 89dd6a53aa0ff700b713b57c5d8d001424557b1d
############################################################################
#  Part  4: Training SVM with a kernel                                     #
#  Select hyperparameters C and sigma                                      #
############################################################################

# load ex4data3.mat

X,y,Xval,yval = utils.loadval_mat('data/ex4data3.mat')

# transform y and yval from [0,1] to [-1,1]

yy = np.ones(y.shape)
yy[y == 0] = -1
yyval = np.ones(yval.shape)
yyval[yval == 0] = -1

# visualize the data

utils.plot_twoclass_data(X,y,'x1', 'x2',['neg','pos'])
plt.savefig('fig5.pdf')

############################################################################
# select hyperparameters C and sigma for this dataset using                #
# Xval and yval                                                            #
############################################################################

<<<<<<< HEAD
Cvals = [0.01,0.03,0.1,0.3,1,3,10,30]
sigma_vals = [0.01,0.03,0.1,0.3,1,3,10,30]

'''
best_C = 0.01
best_sigma = 0.01
'''
=======
Cvals = [0.01,0.03,0.1,0.3,10,30]
sigma_vals = [0.01,0.03,0.1,0.3,10,30]

best_C = 0.01
best_sigma = 0.01
>>>>>>> 89dd6a53aa0ff700b713b57c5d8d001424557b1d

############################################################################
# TODO                                                                     #
# Your code here to select C and sigma with a validation set               #
#   1. do not forget to kernelize X (with an RBF kernel of width sigma)    #
#      before building the classifier with a specific C                    #
#   2. Test each C,sigma combo on the validation data (Xval,yval)          #
#      after transforming Xval to kernel form                              #
############################################################################
# about 15 lines of code expected to get best_C and best_sigma             #
# your code should determine best_C and best_sigma                         #
############################################################################
<<<<<<< HEAD
=======
<<<<<<< HEAD


=======
>>>>>>> 89dd6a53aa0ff700b713b57c5d8d001424557b1d
best_C = None
best_sigma = None
results = {}
best_val = -1
for sigma in sigma_vals:   
        # kernelize X
	K = np.array([utils.gaussian_kernel(x1,x2,sigma) for x1 in X for x2 in X]).reshape(X.shape[0],X.shape[0])
	scaler = preprocessing.StandardScaler().fit(K)
	scaleK = scaler.transform(K)
	KK = np.vstack([np.ones((scaleK.shape[0],)),scaleK]).T
	# kernelize Xval
	Kval = np.array([utils.gaussian_kernel(x1,x2,sigma) for x1 in Xval for x2 in X]).reshape(Xval.shape[0],X.shape[0])
	scaler_val = preprocessing.StandardScaler().fit(Kval)
	scaleKval = scaler_val.transform(Kval)
	KKval = np.vstack([np.ones((scaleKval.shape[0],)),scaleKval.T]).T
	
        for C in Cvals:
		print("Training: C=%e,sigma=%e"%(C,sigma))		
		# set up the SVM and learn the parameters
		svm = LinearSVM_twoclass()
		svm.theta = np.zeros((KK.shape[1],))
		svm.train(KK,yy,learning_rate=1e-4,C=C,num_iters=20000,verbose=False)
		
		# calculate validation accuracy
		va=np.mean(yyval == svm.predict(KKval))
		results[C,sigma]=va
		if va>best_val:
			best_val=va
			best_C=C
			best_sigma = sigma
if(best_C and best_sigma):
        print("*****Problem 3.2******")
        print("Best C = %e Best sigma = %e with validation accuracy = %e"%(best_C,best_sigma,results[best_C,best_sigma]))
<<<<<<< HEAD
=======
>>>>>>> 90033ce... 3.3 completed3.3 completed3.3 completed
>>>>>>> 89dd6a53aa0ff700b713b57c5d8d001424557b1d
############################################################################
#   end of your code                                                       #
############################################################################

# train an SVM on (X,y) with best_C and best_sigma

svm = LinearSVM_twoclass()
K = np.array([utils.gaussian_kernel(x1,x2,best_sigma) for x1 in X for x2 in X]).reshape(X.shape[0],X.shape[0])
scaler = preprocessing.StandardScaler().fit(K)
scaleK = scaler.transform(K)
KK = np.vstack([np.ones((scaleK.shape[0],)),scaleK]).T

svm.theta = np.zeros((KK.shape[1],))
svm.train(KK,yy,learning_rate=1e-4,C=best_C,num_iters=20000,verbose=False)

# visualize the boundary

utils.plot_decision_kernel_boundary(X,y,scaler,best_sigma,svm,'','',['neg','pos'])
plt.savefig("fig6.pdf")
<<<<<<< HEAD

=======
>>>>>>> 89dd6a53aa0ff700b713b57c5d8d001424557b1d
