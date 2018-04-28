#Jonathan Rinnarv - 9301213634 - rinnarv@kth.se

import numpy as np
import math
from matplotlib import pyplot as plt
import sys
import copy

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

	numVal = 1000
	xValidate= xTrain[:,0:numVal]
	YValidate= YTrain[:,0:numVal]
	yValidate= yTrain[0:numVal]

	xTrain= xTrain[:,numVal:]
	YTrain= YTrain[:,numVal:]
	yTrain= yTrain[numVal:]

	xTest, YTest, yTest = loadBatch("test_batch")

	xTrain, xValidate, xTest = normalizationInput(xTrain, xValidate, xTest)

	return xTrain, YTrain, yTrain, xValidate, YValidate, yValidate, xTest, YTest, yTest

def normalizationInput(train, validation, test):
	mean = np.mean(train)
	train -= mean
	validation -= mean
	test -= mean
	return train, validation, test

def getSomeData():
	xTrain, YTrain, yTrain = loadBatch("data_batch_1")
	xValidate, YValidate, yValidate = loadBatch("data_batch_2")
	xTest, YTest, yTest = loadBatch("test_batch")
	xTrain, xValidate, xTest = normalizationInput(xTrain, xValidate, xTest)

	return xTrain, YTrain, yTrain, xValidate, YValidate, yValidate, xTest, YTest, yTest

def getInitData(X,Y, topology, He=False):
	var = 0.001
	if He:
		var = 2/X.shape[0]
	W = []
	b = []
	if(len(topology) != 0):
		W.append(np.matrix([[np.random.normal(0,var) for d in range(X.shape[0])] for K in range(topology[0])]))
		b.append(np.matrix([[np.random.normal(0,var)] for K in range(topology[0])]))
		for l in range(1 , len(topology)):
			W.append(np.matrix([[np.random.normal(0,var) for d in range(topology[l-1])] for K in range(topology[l])]))
			b.append(np.matrix([[np.random.normal(0,var)] for K in range(topology[l])]))
		W.append(np.matrix([[np.random.normal(0,var) for d in range(topology[-1])] for K in range(Y.shape[0])]))
		b.append(np.matrix([[np.random.normal(0,var)] for K in range(Y.shape[0])]))
	else:
		W.append(np.matrix([[np.random.normal(0,var) for d in range(X.shape[0])] for K in range(Y.shape[0])]))
		b.append(np.matrix([[np.random.normal(0,var)] for K in range(Y.shape[0])]))
	return W, b

def evaluateClassifier(X, W, b, muAv, vAv, first=False):
	s = []
	sh = []
	mu = []
	var = []
	Xl = []
	Xl.append(X)
	for l in range(len(W) - 1):
		s.append(W[l]*Xl[l] + b[l])
		mu.append(np.mean(s[l], axis=1))
		var.append(np.var(s[l], axis=1))
		if(first):
			sh.append(batchNormalize(s[l], mu[l], var[l]))
		else:
			sh.append(batchNormalize(s[l], muAv[l], vAv[l]))
		Xl.append(np.maximum(sh[l], 0))
	s.append(W[-1]*Xl[-1] + b[-1])
	sExp = np.exp(s[-1])
	p = sExp/sum(sExp)
	return p, s, Xl, mu, var

def batchNormalize(s, mu, var):
	return np.multiply(np.power(var + 1e-05, -0.5), (s-mu))

def batchNormBackPass(g, s, mu, var):
	Vb = var + 1e-05
	vBSq = np.power(Vb, -0.5)
	sMu = s - mu
	n = s.shape[0]
	gVgSq = np.multiply(g, vBSq)
	gradVar = -0.5 * np.sum(np.multiply(np.multiply(g,np.power(Vb, -3./2)), sMu), axis=1)
	gradMu = - np.sum(gVgSq, axis=1)

	return gVgSq + (2/n) * np.multiply(gradVar, sMu) + gradMu/n

def getLCross(y, P):
	lCross = 0.0
	for i in range(P.shape[1]):
		lCross -= np.log(P[y[i],i])
	return lCross

def computeCost(X, Y, y, W, b, lamda, muAv, vAv):
	P,_,_,_,_ = evaluateClassifier(X,W,b, muAv, vAv)
	lCross = getLCross(y,P)
	L2 = 0
	for l in range(len(W)):
		L2 += np.sum(np.power(W[l],2))
	return (lCross/X.shape[1] + lamda*L2).item(0)

def computeAccuracy(X, y, W, b, muAv, vAv):
	P,_,_,_,_ = evaluateClassifier(X,W,b, muAv, vAv)
	corrects = 0.0
	for i in range(len(y)):
		if y[i] == np.argmax(P[:,i]):
			corrects+=1
	return corrects/len(y)

def computeGradients(X, Y, W, b, lamda, muAv, vAv, a, first):
	gradW = [np.zeros(W[l].shape) for l in range(len(W))]
	gradB = [np.zeros(b[l].shape) for l in range(len(b))]

	P, s, Xl, mu, var = evaluateClassifier(X,W,b, muAv, vAv, first)

	if(first):
		muAv = copy.deepcopy(mu)
		vAv = copy.deepcopy(var)
	else:
		for l in range(len(muAv)):
			muAv[l] = a*muAv[l] + (1 - a)*mu[l]
			vAv = a*vAv[l] + (1 - a)*var[l]

	g = P-Y

	for l in reversed(range(len(W))):
		if l != len(W) -1: 
			g = batchNormBackPass(g, s[l], muAv[l], vAv[l])

		gradB[l] = np.sum(g,axis=1)
		gradW[l] = g*Xl[l].T

		if l > 0:
			g = g.T*W[l]
			ind = (s[l -1] > 0)
			g = np.multiply(g.T,ind)

		gradW[l] /= X.shape[1]
		gradB[l] /= X.shape[1]
		gradW[l] += 2*lamda*W[l]

	return gradW, gradB

def computeGradsNum(X, Y, y, W, b, lamda, h):

	gradW = [np.zeros(W[l].shape) for l in range(len(W))]
	gradB = [np.zeros(b[l].shape) for l in range(len(b))]

	c = computeCost(X, Y, y, W, b, lamda)

	for l in range(len(W)):
		for i in range(b[l].shape[0]):
			bTry = copy.deepcopy(b)
			bTry[l][i] += h
			c2 = computeCost(X, Y, y, W, bTry, lamda)
			gradB[l][i] = (c2-c)/h

		for i in range(W[l].shape[0]):
			for j in range(W[l].shape[1]):
				WTry = copy.deepcopy(W)
				WTry[l][i,j] += h
				c2 = computeCost(X, Y, y, WTry, b, lamda)
				gradW[l][i,j] = (c2-c)/h
				progressPrint(i*W[l].shape[1] + j,W[l].shape[1] * W[l].shape[0])
	sys.stdout.write('\r'+"100%  ")
	sys.stdout.flush()
	print("")

	return gradW, gradB

def updateNetwork(X, Y, GDparams, W, b, lamda, momentumW, momentumB, muAv, vAv, first):
	gradW, gradB = computeGradients(X, Y, W, b, lamda, muAv, vAv, GDparams[5], first)
	for l in range(len(W)):
		momentumW[l] = GDparams[4]*momentumW[l] + GDparams[1]*gradW[l]
		momentumB[l] = GDparams[4]*momentumB[l] + GDparams[1]*gradB[l]
		W[l] -= momentumW[l]
		b[l] -= momentumB[l]


def miniBatchGD(X, Y, y, GDparams, W, b, lamda, XV, YV, yV, momentumW, momentumB, muAv, vAv, earlyStop=False):

	costTrain = [0.0]*GDparams[2]
	accTrain = [0.0]*GDparams[2]
	costVal = [0.0]*GDparams[2]
	accVal = [0.0]*GDparams[2]
	first = True
	stoppedAt = 0
	for epoch in range(GDparams[2]):
		stoppedAt = epoch + 1
		for i in range(1, math.floor(X.shape[1]/GDparams[0])):
			start = (i-1)*GDparams[0]
			end = i*GDparams[0]
			XBatch = X[:,start:end]
			YBatch = Y[:,start:end]
			updateNetwork(XBatch, YBatch, GDparams, W, b, lamda, momentumW, momentumB, muAv, vAv, first)
			if (first):
				first = False
		GDparams[1] *= GDparams[3] #Decay eta
		costTrain[epoch] = computeCost(X, Y, y, W, b, lamda, muAv, vAv)
		accTrain[epoch] = computeAccuracy(X, y, W, b, muAv, vAv)
		costVal[epoch] = computeCost(XV, YV, yV, W, b, lamda, muAv, vAv)
		accVal[epoch] = computeAccuracy(XV, yV, W, b, muAv, vAv)
		if earlyStop and epoch > 5 and (costVal[epoch - 1] - costVal[epoch]) < 0.0001:
			break
		progressPrint(epoch ,GDparams[2])
	sys.stdout.write('\r'+"100%  ")
	sys.stdout.flush()
	print("")

	labels = []
	labels.append(plt.plot(costTrain[0:stoppedAt], label="Training")[0])
	labels.append(plt.plot(costVal[0:stoppedAt], label="Validation")[0])
	plt.legend(handles=labels)
	plt.title("Cross entropy cost")
	plt.ylabel("Cost")
	plt.xlabel("Epoch")
	plt.show()
	labels = []
	labels.append(plt.plot(accTrain[0:stoppedAt], label="Training")[0])
	labels.append(plt.plot(accVal[0:stoppedAt], label="Validation")[0])
	plt.legend(handles=labels)
	plt.title("Cross entropy Accuracy")
	plt.ylabel("Accuracy")
	plt.xlabel("Epoch")
	plt.show()

def checkGradTest():
	X, Y, y = loadBatch("data_batch_1")
	n = 10
	d = 3072 #3072
	X = X[0:d,0:n]
	Y = Y[0:d,0:n]
	y = y[0:n]
	W, b = getInitData(X,Y, [50, 30, 20])
	lamda = 0.0
	analW, analB = computeGradients(X, Y, W, b, lamda)
	numW, numB = computeGradsNum(X, Y, y, W, b, lamda, 1e-05)

	for l in range(len(W)):
		print("------")
		wError = np.max(abs(analW[l] - numW[l]) / np.clip(abs(analW[l]) + abs(numW[l]), a_min=1e-06, a_max=9999))
		bError = np.max(abs(analB[l] - numB[l]) / np.clip(abs(analB[l]) + abs(numB[l]), a_min=1e-06, a_max=9999))
		print("Absolute differance")
		print("W"+str(l)+" = " + str(np.max(abs(analW[l] - numW[l]))))
		print("b"+str(l)+" = " + str(np.max(abs(analB[l] - numB[l]))))
		print("")
		print("Error")
		print("W"+str(l)+" = " + str(wError))
		print("b"+str(l)+" = " + str(bError))

def initMomentum(W, b):
	momentumW = []
	momentumB = []
	for l in range(len(W)):
		momentumW.append(np.zeros(W[l].shape))
		momentumB.append(np.zeros(b[l].shape))

	return momentumW, momentumB

def initBatchNorm(W):
	muAv = []
	vAv = []
	for l in range(len(W) -1):
		muAv.append(np.ones((W[l].shape[0], 1)))
		vAv.append(np.ones((W[l].shape[0], 1)))
	return muAv, vAv

def test():
	X, Y, y, XValidate, YValidate, yValidate, xTest, YTest, yTest = getSomeData()
	lamda = 0.00049
	GDparams = [100, 0.1, 10, 0.95, 0.9, 0.99] #BatchSize, eta, epoch, decay, rho, alpha
	W, b = getInitData(X, Y, [50], He=True)
	momentumW, momentumB = initMomentum(W, b)
	muAv, vAv = initBatchNorm(W)
	miniBatchGD(X, Y, y, GDparams, W, b, lamda, XValidate, YValidate, yValidate, momentumW, momentumB, muAv, vAv, earlyStop=False)
	print(computeAccuracy(xTest, yTest, W, b, muAv, vAv)) 
	

def progressPrint(nominator, denominator):
	denominator = float(denominator)*100
	nominator = float(nominator)*100
	if(nominator % (denominator/100) == 0):
		number = str(round((nominator/denominator)*100, 0))
		sys.stdout.write('\r'+ number + "%")
		sys.stdout.flush()

#checkGradTest()
test()