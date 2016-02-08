import numpy as np
import music_utils
from one_vs_all import one_vs_allLogisticRegressor
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, classification_report

# some global constants

MUSIC_DIR = "music/"
genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

# select the CEPS or FFT representation

X,y = music_utils.read_ceps(genres,MUSIC_DIR)
Xfft,yfft = music_utils.read_fft(genres,MUSIC_DIR)

# select a regularization parameter

reg = 1.0

# create a 1-vs-all classifier

ova_logreg = one_vs_allLogisticRegressor(np.arange(10))
ova_logreg_fft = one_vs_allLogisticRegressor(np.arange(10))

#  divide X into train and test sets

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
Xfft_train, Xfft_test, yfft_train, yfft_test = cross_validation.train_test_split(Xfft, yfft, test_size=0.2)

# train the K classifiers in 1-vs-all mode

ova_logreg.train(X_train,y_train,reg,'l2')
ova_logreg_fft.train(Xfft_train,yfft_train,reg,'l2')
# predict on the set aside test set

ypred = ova_logreg.predict(X_test)
yfft_pred = ova_logreg_fft.predict(Xfft_test)

print "Confusion matrix using Mel Cepstral represenation"
print confusion_matrix(y_test,ypred)

print "Confusion matrix using FFT represenation"
print confusion_matrix(yfft_test,yfft_pred)

# code for Problem 3D2
print "******************** Problem 3D2 ******************** "
cm = confusion_matrix(y_test,ypred)
cmfft = confusion_matrix(yfft_test,yfft_pred)

overall_accuracy = cm.trace()/float(cm.sum())
overall_accuracy_fft = cmfft.trace()/float(cmfft.sum())
print "--- Overall accuracy with Mel Cepstral representation", overall_accuracy
print "--- Overall accuracy with Fourier representation", overall_accuracy_fft

genre_accuracy = np.multiply(cm.diagonal() , 1.0/cm.sum(1))
genre_accuracy_fft = np.multiply(cmfft.diagonal() , 1.0/cmfft.sum(1))

for i in range(0,10):
    print "--- Genre = ", genres[i], " accuracy with Mel Cepstral representation", genre_accuracy[i]
    print "--- Genre = ", genres[i], " accuracy with Fourier representation", genre_accuracy_fft[i]