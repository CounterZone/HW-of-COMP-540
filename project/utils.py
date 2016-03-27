import cPickle
import numpy as np
labels=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
def load_train():
	x=cPickle.load(open("data/train/train_1","rb"))
	for i in range(2,6):
		x=np.vstack([x,cPickle.load(open("data/train/train_%d"%i,"rb"))])
	y=cPickle.load(open("data/train/labels","rb"))
	return x,y
def load_test():
	x=cPickle.load(open("data/test/test_1","rb"))
	for i in range(2,31):
		x=np.vstack([x,cPickle.load(open("data/test/test_%d"%i,"rb"))])
	return x
def output(y,fname="predict.csv"):
	f=open(fname,"w")
	for i in range(len(y)):
		f.write("%d,"%i+labels[y[i]]+"\n")
	f.close()