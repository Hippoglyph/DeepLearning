#Jonathan Rinnarv - 9301213634 - rinnarv@kth.se

import numpy as np
import math
from matplotlib import pyplot as plt
import sys

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
	'''
	n = 1000
	nO = int(n/4)
	xTrain = xTrain[:,0:n]
	YTrain = YTrain[:,0:n]
	yTrain = yTrain[0:n]
	'''
	xValidate, YValidate, yValidate = loadBatch("data_batch_2")
	'''
	xValidate = xValidate[:,0:nO]
	YValidate = YValidate[:,0:nO]
	yValidate = yValidate[0:nO]
	'''
	xTest, YTest, yTest = loadBatch("test_batch")
	'''
	xTest = xTest[:,0:nO]
	YTest = YTest[:,0:nO]
	yTest = yTest[0:nO]
	'''
	xTrain, xValidate, xTest = normalizationInput(xTrain, xValidate, xTest)

	return xTrain, YTrain, yTrain, xValidate, YValidate, yValidate, xTest, YTest, yTest

def getInitData(X,Y, hiddenNumber, Xavier=False):
	var = 0.001
	if Xavier:
		var = 1/X.shape[0]
	W1 = np.matrix([[np.random.normal(0,var) for d in range(X.shape[0])] for K in range(hiddenNumber)])
	b1 = np.matrix([[np.random.normal(0,var)] for K in range(hiddenNumber)])
	W2 = np.matrix([[np.random.normal(0,var) for d in range(hiddenNumber)] for K in range(Y.shape[0])])
	b2 = np.matrix([[np.random.normal(0,var)] for K in range(Y.shape[0])])
	return W1, b1, W2, b2

def evaluateClassifier(X,W1,b1,W2,b2):
	s1 = W1*X + b1
	h = np.maximum(s1, 0)
	s = W2*h + b2
	sExp = np.exp(s)
	p = sExp/sum(sExp)
	return p, h, s1

def getLCross(y, P):
	lCross = 0.0
	for i in range(P.shape[1]):
		lCross -= np.log(P[y[i],i])
	return lCross

def computeCost(X, Y, y, W1, b1, W2, b2, lamda):
	P,_,_ = evaluateClassifier(X,W1,b1,W2,b2)
	lCross = getLCross(y,P)
	L2 = np.sum(np.power(W1,2))+np.sum(np.power(W2,2))
	return (lCross/X.shape[1] + lamda*L2).item(0)

def computeAccuracy(X, y, W1, b1, W2, b2):
	P,_,_ = evaluateClassifier(X,W1,b1,W2,b2)
	corrects = 0.0
	for i in range(len(y)):
		if y[i] == np.argmax(P[:,i]):
			corrects+=1
	return corrects/len(y)

def computeGradients(X, Y, W1, b1, W2, b2, lamda):
	P, h, s1 = evaluateClassifier(X,W1,b1,W2,b2)
	g = P-Y
	gradB1 = np.zeros(b1.shape)
	gradW1 = np.zeros(W1.shape)
	gradB2 = np.zeros(b2.shape)
	gradW2 = np.zeros(W2.shape)

	for i in range(g.shape[1]):
		gradB2 += g[:,i]
		gradW2 += g[:,i]*h[:,i].T

	g = g.T*W2
	ind = np.where(s1>0, 1, 0)
	for i in range(g.shape[0]):
		g2 = (g[i] * np.diag(ind[:,i])).T
		gradB1 += g2
		gradW1 += g2*X[:,i].T

	gradB1 /= X.shape[1]
	gradW1 /= X.shape[1]
	gradW1 += 2*lamda*W1
	gradB2 /= X.shape[1]
	gradW2 /= X.shape[1]
	gradW2 += 2*lamda*W2
	return gradW1, gradB1, gradW2, gradB2

def computeGradsNum(X, Y, y, W1, b1, W2, b2, lamda, h):

	gradW1 = np.zeros(W1.shape)
	gradB1 = np.zeros(b1.shape)
	gradW2 = np.zeros(W2.shape)
	gradB2 = np.zeros(b2.shape)

	c = computeCost(X, Y, y, W1, b1, W2, b2, lamda)

	print("B1")
	for i in range(b1.shape[0]):
		bTry = b1.copy()
		bTry[i] += h
		c21 = computeCost(X, Y, y, W1, bTry, W2, b2, lamda)
		gradB1[i] = (c21-c)/h
		progressPrint(i,b1.shape[0])
	sys.stdout.write('\r'+"100%  ")
	sys.stdout.flush()
	print("")

	print("B2")
	for i in range(b2.shape[0]):
		bTry = b2.copy()
		bTry[i] += h
		c22 = computeCost(X, Y, y, W1, b1, W2, bTry, lamda)
		gradB2[i] = (c22-c)/h
		progressPrint(i,b2.shape[0])
	sys.stdout.write('\r'+"100%  ")
	sys.stdout.flush()
	print("")

	print("W1")
	for i in range(W1.shape[0]):
		for j in range(W1.shape[1]):
			WTry = W1.copy()
			WTry[i,j] += h
			c21 = computeCost(X, Y, y, WTry, b1, W2, b2, lamda)
			gradW1[i,j] = (c21-c)/h
			progressPrint(i*W1.shape[1] + j,W1.shape[1] * W1.shape[0])
	sys.stdout.write('\r'+"100%  ")
	sys.stdout.flush()
	print("")

	print("W2")
	for i in range(W2.shape[0]):
		for j in range(W2.shape[1]):
			WTry = W2.copy()
			WTry[i,j] += h
			c22 = computeCost(X, Y, y, W1, b1, WTry, b2, lamda)
			gradW2[i,j] = (c22-c)/h
			progressPrint(i*W2.shape[1] + j,W2.shape[1] * W2.shape[0])
	sys.stdout.write('\r'+"100%  ")
	sys.stdout.flush()
	print("")

	return gradW1, gradB1, gradW2, gradB2

def updateNetwork(X, Y, GDparams, W1, b1, W2, b2, lamda, momentum):
	gradW1, gradB1, gradW2, gradB2 = computeGradients(X, Y, W1, b1, W2, b2, lamda)
	momentum[0] = GDparams[4]*momentum[0] + GDparams[1]*gradW1
	momentum[1] = GDparams[4]*momentum[1] + GDparams[1]*gradB1
	momentum[2] = GDparams[4]*momentum[2] + GDparams[1]*gradW2
	momentum[3] = GDparams[4]*momentum[3] + GDparams[1]*gradB2
	W1 -= momentum[0]
	b1 -= momentum[1]
	W2 -= momentum[2]
	b2 -= momentum[3]

def miniBatchGD(X, Y, y, GDparams, W1, b1, W2, b2, lamda, XV, YV, yV, momentum, earlyStop=False):

	costTrain = [0.0]*GDparams[2]
	accTrain = [0.0]*GDparams[2]
	costVal = [0.0]*GDparams[2]
	accVal = [0.0]*GDparams[2]

	stoppedAt = 0
	for epoch in range(GDparams[2]):
		stoppedAt = epoch + 1
		for i in range(1, math.floor(X.shape[1]/GDparams[0])):
			start = (i-1)*GDparams[0]
			end = i*GDparams[0]
			XBatch = X[:,start:end]
			YBatch = Y[:,start:end]
			updateNetwork(XBatch, YBatch, GDparams, W1, b1, W2, b2, lamda, momentum)
		GDparams[1] *= GDparams[3] #Decay eta
		costTrain[epoch] = computeCost(X, Y, y, W1, b1, W2, b2, lamda)
		accTrain[epoch] = computeAccuracy(X, y, W1, b1, W2, b2)
		costVal[epoch] = computeCost(XV, YV, yV, W1, b1, W2, b2, lamda)
		accVal[epoch] = computeAccuracy(XV, yV, W1, b1, W2, b2)
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
	d = 3072
	X = X[0:d,0:n]
	Y = Y[0:d,0:n]
	y = y[0:n]
	W1, b1, W2, b2 = getInitData(X,Y, 50)
	lamda = 0.0
	analW1, analB1, analW2, analB2 = computeGradients(X, Y, W1, b1, W2, b2, lamda)
	numW1, numB1, numW2, numB2 = computeGradsNum(X, Y, y, W1, b1, W2, b2, lamda, 1e-05)

	w1Error = np.max(abs(analW1 - numW1) / np.clip(abs(analW1) + abs(numW1), a_min=1e-06, a_max=9999))
	b1Error = np.max(abs(analB1 - numB1) / np.clip(abs(analB1) + abs(numB1), a_min=1e-06, a_max=9999))
	w2Error = np.max(abs(analW2 - numW2) / np.clip(abs(analW2) + abs(numW2), a_min=1e-06, a_max=9999))
	b2Error = np.max(abs(analB2 - numB2) / np.clip(abs(analB2) + abs(numB2), a_min=1e-06, a_max=9999))
	print("W1 = " + str(w1Error))
	print("b1 = " + str(b1Error))
	print("W2 = " + str(w2Error))
	print("b2 = " + str(b2Error))

	print("W1 = " + str(np.max(abs(analW1 - numW1))))
	print("b1 = " + str(np.max(abs(analB1 - numB1))))
	print("W2 = " + str(np.max(abs(analW2 - numW2))))
	print("b2 = " + str(np.max(abs(analB2 - numB2))))

def initMomentum(W1, b1, W2, b2):
	W1M = np.zeros(W1.shape)
	W2M = np.zeros(W2.shape)
	b1M = np.zeros(b1.shape)
	b2M = np.zeros(b2.shape)
	return [W1M, b1M, W2M, b2M]

def fit(X, Y, y, GDparams, W1, b1, W2, b2, lamda, momentum):
	for epoch in range(GDparams[2]):
		stoppedAt = epoch + 1
		for i in range(1, math.floor(X.shape[1]/GDparams[0])):
			start = (i-1)*GDparams[0]
			end = i*GDparams[0]
			XBatch = X[:,start:end]
			YBatch = Y[:,start:end]
			updateNetwork(XBatch, YBatch, GDparams, W1, b1, W2, b2, lamda, momentum)
		GDparams[1] *= GDparams[3] #Decay eta

def parameterTest():

	nIters = 100

	valAcc = [0.0]*nIters
	parameters  = [0.0, 0.0]*nIters
	bestAcc = [0,0,0]
	bestId = [0,0,0]

	GDparams = [100, 0, 10, 0.95, 0.9] #BatchSize, eta, epoch, decay, rho

	X, Y, y, XValidate, YValidate, yValidate, xTest, YTest, yTest = getSomeData()

	for i in range(nIters):
		eta = 10**((-3 + (-1 + 3)*np.random.random()))
		lamda = 10**((-4 + (-1 + 4)*np.random.random()))
		GDparams[1] = eta

		W1, b1, W2, b2 = getInitData(X, Y, 50, Xavier=True)
		momentum = initMomentum(W1, b1, W2, b2)

		fit(X, Y, y, GDparams, W1, b1, W2, b2, lamda, momentum)
		valAcc[i] = computeAccuracy(XValidate, yValidate, W1, b1, W2, b2)
		parameters[i] = [lamda, eta]
		progressPrint(i , nIters)
	sys.stdout.write('\r'+"100%  ")
	sys.stdout.flush()
	print("")

	for i in range(nIters):
		argMin = np.argmin(bestAcc)
		if valAcc[i] > bestAcc[argMin]:
			bestAcc[argMin] = valAcc[i]
			bestId[argMin] = i

	with open("parameters", "w") as f:
		for i in range(nIters):
			addOn = ""
			if i in bestId:
				addOn = " < Done good"
			f.write("Accuarcy: " + "%.3f" % round(valAcc[i], 3) + " \t Lambda: " + "%.5f" % round(parameters[i][0], 5) + " \t Eta: " + "%.5f" % round(parameters[i][1], 5) + addOn +"\n")

def fineSearch():
	nIters = 100

	valAcc = [0.0]*nIters
	parameters  = [0.0, 0.0]*nIters
	bestAcc = [0,0,0]
	bestId = [0,0,0]

	GDparams = [100, 0, 10, 0.95, 0.9] #BatchSize, eta, epoch, decay, rho

	X, Y, y, XValidate, YValidate, yValidate, xTest, YTest, yTest = getSomeData()

	for i in range(nIters):
		eta = ((0.015 + (0.055 - 0.015)*np.random.random()))
		lamda = ((0.0001 + (0.002 - 0.0001)*np.random.random()))
		GDparams[1] = eta

		W1, b1, W2, b2 = getInitData(X, Y, 50, Xavier=True)
		momentum = initMomentum(W1, b1, W2, b2)

		fit(X, Y, y, GDparams, W1, b1, W2, b2, lamda, momentum)
		valAcc[i] = computeAccuracy(XValidate, yValidate, W1, b1, W2, b2)
		parameters[i] = [lamda, eta]
		progressPrint(i , nIters)
	sys.stdout.write('\r'+"100%  ")
	sys.stdout.flush()
	print("")

	for i in range(nIters):
		argMin = np.argmin(bestAcc)
		if valAcc[i] > bestAcc[argMin]:
			bestAcc[argMin] = valAcc[i]
			bestId[argMin] = i

	with open("fineSearch", "w") as f:
		for i in range(nIters):
			addOn = ""
			if i in bestId:
				addOn = " < Done good"
			f.write("Accuarcy: " + "%.3f" % round(valAcc[i], 3) + " \t Lambda: " + "%.5f" % round(parameters[i][0], 5) + " \t Eta: " + "%.5f" % round(parameters[i][1], 5) + addOn +"\n")

def test():
	lamda = 0.1
	GDparams = [10, 0.1, 5, 0.95, 0.9] #BatchSize, eta, epoch, decay, rho
	X, Y, y, XValidate, YValidate, yValidate, xTest, YTest, yTest = getSomeData()
	W1, b1, W2, b2 = getInitData(X, Y, 50, Xavier=True)
	momentum = initMomentum(W1, b1, W2, b2)
	miniBatchGD(X, Y, y, GDparams, W1, b1, W2, b2, lamda, XValidate, YValidate, yValidate, momentum, earlyStop=False)
	print(computeAccuracy(xTest, yTest, W1, b1, W2, b2))

def progressPrint(nominator, denominator):
	denominator = float(denominator)*100
	nominator = float(nominator)*100
	if(nominator % (denominator/100) == 0):
		number = str(round((nominator/denominator)*100, 0))
		sys.stdout.write('\r'+ number + "%")
		sys.stdout.flush()

#checkGradTest()
#test()
#parameterTest()
fineSearch()