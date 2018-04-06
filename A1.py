import numpy as np
import math
from matplotlib import pyplot as plt

def unpickle(file):
	import pickle
	with open("cifar-10-batches-py/"+file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def imgShow(x):
	data = x.reshape(3,32,32).transpose([1,2,0])
	fig = plt.figure(figsize = (1,1))
	ax = fig.add_subplot(111)
	ax.imshow(data, interpolation='gaussian')
	plt.axis('off')
	plt.show()

def loadBatch(file):
	pickle = unpickle(file)
	X = pickle[b"data"].T
	labels = np.array(pickle[b"labels"])
	Y = np.zeros((10,len(labels)))
	for i in range(Y.shape[1]):
		Y[labels[i]][i] = 1
	return X/255,Y,labels

def getAllData():
	xTrain1, YTrain1, yTrain1 = loadBatch("data_batch_1")
	xTrain2, YTrain2, yTrain2 = loadBatch("data_batch_2")
	xTrain3, YTrain3, yTrain3 = loadBatch("data_batch_3")
	xTrain4, YTrain4, yTrain4 = loadBatch("data_batch_4")
	xTrain5, YTrain5, yTrain5 = loadBatch("data_batch_5")

	xTrain = np.concatenate((xTrain1,xTrain2,xTrain3,xTrain4,xTrain5), axis=1)
	YTrain = np.concatenate((YTrain1,YTrain2,YTrain3,YTrain4,YTrain5), axis=1)
	yTrain = np.concatenate((yTrain1,yTrain2,yTrain3,yTrain4,yTrain5), axis=0)

	xValidate, YValidate, yValidate = loadBatch("test_batch")

	return xTrain, YTrain, yTrain, xValidate, YValidate, yValidate

def getInitData(X,Y):
	W = np.matrix([[np.random.normal(0,0.01) for d in range(X.shape[0])] for K in range(Y.shape[0])])
	b = np.matrix([[np.random.normal(0,0.01)] for K in range(Y.shape[0])])
	return W, b

def evaluateClassifier(X,W,b):
	s = W*X + b
	sExp = np.exp(s)
	p = sExp/sum(sExp)
	return p

def getLCross(Y, P):
	lCross = 0.0
	for i in range(Y.shape[1]):
		lCross += -np.log(Y[:,i]*P[:,i])
	return lCross

def computeCost(X, Y, W, b, lamda):
	P = evaluateClassifier(X,W,b)
	lCross = getLCross(Y,P)
	L2 = 0.0
	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			L2 += W[i,j]**2
	return (lCross/X.shape[1] + lamda*L2).item(0)

def computeAccuracy(X, y, W, b):
	P = evaluateClassifier(X,W,b)
	corrects = 0.0
	for i in range(len(y)):
		if y[i] == np.argmax(P[:,i]):
			corrects+=1
	return corrects/len(y)

def computeGradients(X, Y, W, lamda):
	P = evaluateClassifier(X,W,b)
	g = -(Y-P)
	gradB = np.zeros((Y.shape[0], 1))
	gradW = np.zeros(W.shape)
	for i in range(g.shape[1]):
		gradB += g[:,i]
		gradW += g[:,i]*X[:,i].T
	gradB /= X.shape[1]
	gradW /= X.shape[1]
	gradW += 2*lamda*W
	return gradW, gradB

def updateNetwork(X, Y, GDparams, W, b, lamda):
	gradW, gradB = computeGradients(X, Y, W, lamda)
	W -= GDparams[1]*gradW
	b -= GDparams[1]*gradB

def miniBatchGD(X, Y, y, GDparams, W, b, lamda, XV, YV, yV):

	costTrain = [0.0]*GDparams[2]
	accTrain = [0.0]*GDparams[2]
	costVal = [0.0]*GDparams[2]
	accVal = [0.0]*GDparams[2]

	for epoch in range(GDparams[2]):
		print(epoch)
		for i in range(1, math.floor(len(X)/GDparams[0])):
			start = (i-1)*GDparams[0]
			end = i*GDparams[0]
			XBatch = X[:,start:end]
			YBatch = Y[:,start:end]
			updateNetwork(XBatch, YBatch, GDparams, W, b, lamda)
		costTrain[epoch] = computeCost(X, Y, W, b, lamda)
		accTrain[epoch] = computeAccuracy(X, y, W, b)
		costVal[epoch] = computeCost(XV, YV, W, b, lamda)
		accVal[epoch] = computeAccuracy(XV, yV, W, b)
	plt.plot(costTrain)
	plt.plot(costVal)
	plt.show()
	plt.plot(accTrain)
	plt.plot(accVal)
	plt.show()

def printW(W):
	fig, ax = plt.subplots(1, W.shape[0])
	for k in range(W.shape[0]):
		img = np.array(W[k]).reshape(3,32,32).transpose([1,2,0])
		minImg = img.min()
		maxImg = img.max()
		img = (img - minImg)/(maxImg - minImg)
		ax[k].imshow(img, interpolation='gaussian')
		ax[k].set_xticks(())
		ax[k].set_yticks(())
	plt.show()

lamda = 0.0
GDparams = [100, 0.01, 40]
X, Y, y, XValidate, YValidate, yValidate = getAllData()
W, b = getInitData(X,Y)
miniBatchGD(X, Y, y, GDparams, W, b, lamda, XValidate, YValidate, yValidate)
#printW(W)

#imgShow(file[b"data"][np.random.randint(0, len(file[b"data"]))].T)