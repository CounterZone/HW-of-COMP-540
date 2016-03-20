from sklearn import preprocessing, metrics
from sklearn.cross_validation import *
import utils
import scipy.io
import numpy as np
from linear_classifier import LinearSVM_twoclass
import matplotlib.pyplot as plt

#############################################################################
# load the SPAM email training and test dataset                             #
#############################################################################

X,y = utils.load_mat('data/spamTrain.mat')
yy = np.ones(y.shape)
yy[y==0] = -1
test_data = scipy.io.loadmat('data/spamTest.mat')
X_test = test_data['Xtest']
y_test = test_data['ytest'].flatten()

#############################################################################
# your code for setting up the best SVM classifier for this dataset         #
# Design the training parameters for the SVM.                               #
# What should the learning_rate be? What should C be?                       #
# What should num_iters be? Should X be scaled? Should X be kernelized?     #
#############################################################################
# your experiments below

svm = LinearSVM_twoclass()
svm.theta = np.zeros((X.shape[1],))
# Experiment to find best C ,learning rate and number of iterations (no kernal)

# X_train, X_val, y_train, y_val = train_test_split(X, yy, test_size=0.2)
# iters=list(np.array(range(801))*5)
# trace={}
# for lr in [0.01,0.05,0.1,0.5]:
# 	trace[lr]=[]
# 	svm.theta = np.zeros((X.shape[1],))
# 	trace[lr].append(metrics.accuracy_score(y_val,svm.predict(X_val)))
# 	for i in range(0,800):
# 		svm.train(X_train,y_train,learning_rate=lr,C=0.1,num_iters=5,verbose=False)
# 		a=metrics.accuracy_score(y_val,svm.predict(X_val))
# 		trace[lr].append(a)
# 		print("%.2f,%d,%f\d"%(lr,i,a))
# 	plt.plot(iters,trace[lr],label="lr=%.2f"%lr)
# plt.xlabel("iterations")
# plt.ylabel("Accurancy on validation set")
# plt.legend(bbox_to_anchor=(0.9, 0.9))
# plt.savefig("no_kernal_lr.pdf")

# Experiment to find best sigma(kernalized)
X_train, X_val, y_train, y_val = train_test_split(X, yy, test_size=0.2)

iters=list(np.array(range(1601))*5)
trace={}
for sigma in [10]:
	trace[sigma]=[]
	K=metrics.pairwise.rbf_kernel(X_train,X_train,1/sigma**2)
	scaler = preprocessing.StandardScaler().fit(K)
	scaleK = scaler.transform(K)
	KK = np.vstack([np.ones((scaleK.shape[0],)),scaleK]).T
	Kval=metrics.pairwise.rbf_kernel(X_val,X_train,1/sigma**2)
	scaler_val = preprocessing.StandardScaler().fit(Kval)
	scaleKval = scaler_val.transform(Kval)
	KKval = np.vstack([np.ones((scaleKval.shape[0],)),scaleKval.T]).T
	svm.theta = np.zeros((KK.shape[1],))
 	trace[sigma].append(metrics.accuracy_score(y_val,svm.predict(KKval)))
	for i in range(0,1600):
		svm.train(KK,y_train,learning_rate=0.01,C=2,num_iters=5,verbose=True)
 		a=metrics.accuracy_score(y_train,svm.predict(KK))
		b=metrics.accuracy_score(y_val,svm.predict(KKval))
		trace[sigma].append(a)
 		print("%.2f,%d,%f,%f\d"%(sigma,i,a,b))
	plt.plot(iters,trace[sigma],label="sigma=%.2f"%sigma)
plt.xlabel("iterations")
plt.ylabel("Accurancy on validation set")
plt.legend(bbox_to_anchor=(0.9, 0.9))
plt.savefig("kernal_sigma.pdf")


#svm.train(X,yy,learning_rate=0.1,C=0.1,num_iters=2000,verbose=True)


#############################################################################
#  end of your code                                                         #
#############################################################################

#############################################################################
# what is the accuracy of the best model on the training data itself?       #
#############################################################################
# 2 lines of code expected

y_pred = svm.predict(X)
print "Accuracy of model on training data is: ", metrics.accuracy_score(yy,y_pred)


#############################################################################
# what is the accuracy of the best model on the test data?                  #
#############################################################################
# 2 lines of code expected


yy_test = np.ones(y_test.shape)
yy_test[y_test==0] = -1
test_pred = svm.predict(X_test)
print "Accuracy of model on test data is: ", metrics.accuracy_score(yy_test,test_pred)


#############################################################################
# Interpreting the coefficients of an SVM                                   #
# which words are the top predictors of spam?                               #
#############################################################################
# 4 lines of code expected

words, inv_words = utils.get_vocab_dict()

index = np.argsort(svm.theta)[-15:]
print "Top 15 predictors of spam are: "
for i in range(-1,-16,-1):
    print words[index[i]+1]


