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
	xValidate, YValidate, yValidate = loadBatch("data_batch_2")
	xTest, YTest, yTest = loadBatch("test_batch")
	xTrain, xValidate, xTest = normalizationInput(xTrain, xValidate, xTest)

	return xTrain, YTrain, yTrain, xValidate, YValidate, yValidate, xTest, YTest, yTest

def getInitData(X,Y, hiddenNumber, He=False):
	var = 0.001
	if He:
		var = 2/X.shape[0]
	W1 = np.matrix([[np.random.normal(0,var) for d in range(X.shape[0])] for K in range(hiddenNumber)])
	b1 = np.matrix([[np.random.normal(0,var)] for K in range(hiddenNumber)])
	W2 = np.matrix([[np.random.normal(0,var) for d in range(hiddenNumber)] for K in range(Y.shape[0])])
	b2 = np.matrix([[np.random.normal(0,var)] for K in range(Y.shape[0])])
	return W1, b1, W2, b2

def evaluateClassifier(X,W1,b1,W2,b2, Sigmoid=False):
	s1 = W1*X + b1
	if (Sigmoid):
		h = sigmoid(s1)
	else:
		h = np.maximum(s1, 0)
	s = W2*h + b2
	sExp = np.exp(s)
	p = sExp/sum(sExp)
	return p, h, s1

def sigmoid(X):
	return 1/(1+np.exp(-X))

def sigDerivative(X):
	newX = np.array(X)
	return newX*(1-newX)

def getLCross(y, P):
	lCross = 0.0
	for i in range(P.shape[1]):
		lCross -= np.log(P[y[i],i])
	return lCross

def computeCost(X, Y, y, W1, b1, W2, b2, lamda, Sigmoid=False):
	P,_,_ = evaluateClassifier(X,W1,b1,W2,b2,Sigmoid)
	lCross = getLCross(y,P)
	L2 = np.sum(np.power(W1,2))+np.sum(np.power(W2,2))
	return (lCross/X.shape[1] + lamda*L2).item(0)

def computeAccuracy(X, y, W1, b1, W2, b2, Sigmoid=False):
	P,_,_ = evaluateClassifier(X,W1,b1,W2,b2,Sigmoid)
	corrects = 0.0
	for i in range(len(y)):
		if y[i] == np.argmax(P[:,i]):
			corrects+=1
	return corrects/len(y)

def computeGradients(X, Y, W1, b1, W2, b2, lamda, Sigmoid=False):
	P, h, s1 = evaluateClassifier(X,W1,b1,W2,b2, Sigmoid)
	g = P-Y
	gradB1 = np.zeros(b1.shape)
	gradW1 = np.zeros(W1.shape)
	gradB2 = np.zeros(b2.shape)
	gradW2 = np.zeros(W2.shape)

	for i in range(g.shape[1]):
		gradB2 += g[:,i]
		gradW2 += g[:,i]*h[:,i].T

	g = W2.T*g
	if (Sigmoid):
		deltaSig = sigDerivative(h)
		for i in range(g.shape[1]):
			g2 = (g[:,i].T * np.diag(deltaSig[:,i])).T
			gradB1 += g2
			gradW1 += g2*X[:,i].T
	else:
		ind = np.where(s1>0, 1, 0)
		for i in range(g.shape[1]):
			g2 = (g[:,i].T * np.diag(ind[:,i])).T
			gradB1 += g2
			gradW1 += g2*X[:,i].T

	gradB1 /= X.shape[1]
	gradW1 /= X.shape[1]
	gradW1 += 2*lamda*W1
	gradB2 /= X.shape[1]
	gradW2 /= X.shape[1]
	gradW2 += 2*lamda*W2
	return gradW1, gradB1, gradW2, gradB2

def computeGradsNum(X, Y, y, W1, b1, W2, b2, lamda, h, Sigmoid=False):

	gradW1 = np.zeros(W1.shape)
	gradB1 = np.zeros(b1.shape)
	gradW2 = np.zeros(W2.shape)
	gradB2 = np.zeros(b2.shape)

	c = computeCost(X, Y, y, W1, b1, W2, b2, lamda, Sigmoid)

	print("B1")
	for i in range(b1.shape[0]):
		bTry = b1.copy()
		bTry[i] += h
		c21 = computeCost(X, Y, y, W1, bTry, W2, b2, lamda, Sigmoid)
		gradB1[i] = (c21-c)/h
		progressPrint(i,b1.shape[0])
	sys.stdout.write('\r'+"100%  ")
	sys.stdout.flush()
	print("")

	print("B2")
	for i in range(b2.shape[0]):
		bTry = b2.copy()
		bTry[i] += h
		c22 = computeCost(X, Y, y, W1, b1, W2, bTry, lamda, Sigmoid)
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
			c21 = computeCost(X, Y, y, WTry, b1, W2, b2, lamda, Sigmoid)
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
			c22 = computeCost(X, Y, y, W1, b1, WTry, b2, lamda, Sigmoid)
			gradW2[i,j] = (c22-c)/h
			progressPrint(i*W2.shape[1] + j,W2.shape[1] * W2.shape[0])
	sys.stdout.write('\r'+"100%  ")
	sys.stdout.flush()
	print("")

	return gradW1, gradB1, gradW2, gradB2

def updateNetwork(X, Y, GDparams, W1, b1, W2, b2, lamda, momentum, Sigmoid=False):
	gradW1, gradB1, gradW2, gradB2 = computeGradients(X, Y, W1, b1, W2, b2, lamda, Sigmoid)
	momentum[0] = GDparams[4]*momentum[0] + GDparams[1]*gradW1
	momentum[1] = GDparams[4]*momentum[1] + GDparams[1]*gradB1
	momentum[2] = GDparams[4]*momentum[2] + GDparams[1]*gradW2
	momentum[3] = GDparams[4]*momentum[3] + GDparams[1]*gradB2
	W1 -= momentum[0]
	b1 -= momentum[1]
	W2 -= momentum[2]
	b2 -= momentum[3]

def miniBatchGD(X, Y, y, GDparams, W1, b1, W2, b2, lamda, XV, YV, yV, momentum, earlyStop=False, Sigmoid=False):

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
			updateNetwork(XBatch, YBatch, GDparams, W1, b1, W2, b2, lamda, momentum, Sigmoid)
		GDparams[1] *= GDparams[3] #Decay eta
		costTrain[epoch] = computeCost(X, Y, y, W1, b1, W2, b2, lamda, Sigmoid)
		accTrain[epoch] = computeAccuracy(X, y, W1, b1, W2, b2, Sigmoid)
		costVal[epoch] = computeCost(XV, YV, yV, W1, b1, W2, b2, lamda, Sigmoid)
		accVal[epoch] = computeAccuracy(XV, yV, W1, b1, W2, b2, Sigmoid)
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
	analW1, analB1, analW2, analB2 = computeGradients(X, Y, W1, b1, W2, b2, lamda, Sigmoid=True)
	numW1, numB1, numW2, numB2 = computeGradsNum(X, Y, y, W1, b1, W2, b2, lamda, 1e-05, Sigmoid=True)

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

def fit(X, Y, y, GDparams, W1, b1, W2, b2, lamda, momentum, Sigmoid=False):
	for epoch in range(GDparams[2]):
		stoppedAt = epoch + 1
		for i in range(1, math.floor(X.shape[1]/GDparams[0])):
			start = (i-1)*GDparams[0]
			end = i*GDparams[0]
			XBatch = X[:,start:end]
			YBatch = Y[:,start:end]
			updateNetwork(XBatch, YBatch, GDparams, W1, b1, W2, b2, lamda, momentum,Sigmoid)
		GDparams[1] *= GDparams[3] #Decay eta

def parameterTest(e_min, e_max, l_min, l_max, fileName, Sigmoid=False):

	nIters = 100

	valAcc = [0.0]*nIters
	parameters  = [0.0, 0.0]*nIters
	bestAcc = [0,0,0]
	bestId = [0,0,0]

	GDparams = [100, 0, 10, 0.95, 0.9] #BatchSize, eta, epoch, decay, rho

	X, Y, y, XValidate, YValidate, yValidate, xTest, YTest, yTest = getSomeData()

	for i in range(nIters):
		eta = 10**((e_min + (e_max - e_min)*np.random.random()))
		lamda = 10**((l_min + (l_max - l_min)*np.random.random()))
		GDparams[1] = eta

		W1, b1, W2, b2 = getInitData(X, Y, 50, He=True)
		momentum = initMomentum(W1, b1, W2, b2)

		fit(X, Y, y, GDparams, W1, b1, W2, b2, lamda, momentum, Sigmoid)
		valAcc[i] = computeAccuracy(XValidate, yValidate, W1, b1, W2, b2, Sigmoid)
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

	with open(filename, "w") as f:
		for i in range(nIters):
			addOn = ""
			if i in bestId:
				addOn = " < Done good"
			f.write("Accuracy: " + "%.3f" % round(valAcc[i], 3) + " \t Lambda: " + "%.5f" % round(parameters[i][0], 5) + " \t Eta: " + "%.5f" % round(parameters[i][1], 5) + addOn +"\n")

def test():
	X, Y, y, XValidate, YValidate, yValidate, xTest, YTest, yTest = getSomeData()
	lamda = 0.00049 #Best lambda 0.00049 Eta 0.02573
	GDparams = [100, 0.02573, 20, 0.95, 0.9] #BatchSize, eta, epoch, decay, rho
	W1, b1, W2, b2 = getInitData(X, Y, 50, He=True)
	momentum = initMomentum(W1, b1, W2, b2)
	miniBatchGD(X, Y, y, GDparams, W1, b1, W2, b2, lamda, XValidate, YValidate, yValidate, momentum, earlyStop=False)
	print(computeAccuracy(xTest, yTest, W1, b1, W2, b2))

def testSig():
	X, Y, y, XValidate, YValidate, yValidate, xTest, YTest, yTest = getAllData()
	lamda = 0.00017 #Best lambda 0.00017 Eta 0.08956
	GDparams = [100, 0.08956, 40, 0.95, 0.9] #BatchSize, eta, epoch, decay, rho
	W1, b1, W2, b2 = getInitData(X, Y, 50, He=True)
	momentum = initMomentum(W1, b1, W2, b2)
	miniBatchGD(X, Y, y, GDparams, W1, b1, W2, b2, lamda, XValidate, YValidate, yValidate, momentum, earlyStop=False, Sigmoid=True)
	print(computeAccuracy(xTest, yTest, W1, b1, W2, b2, Sigmoid=True))

def progressPrint(nominator, denominator):
	denominator = float(denominator)*100
	nominator = float(nominator)*100
	if(nominator % (denominator/100) == 0):
		number = str(round((nominator/denominator)*100, 0))
		sys.stdout.write('\r'+ number + "%")
		sys.stdout.flush()

#checkGradTest()
test()
#testSig()
#parameterTest(-3, -1, -4, -1, "parameters")
#parameterTest(-1.8, -1.25, -4, -2.7, "fineSearch")
#parameterTest(-3, -1, -4, -1, "parametersSigmoid", Sigmoid=True)

#Best lambda  = 0.00049
#Best Eta = 0.02573

#150 hidden 51.98 %
#Best 51.27 %
#Best sigmoid 50.84 %