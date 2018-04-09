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

	numVal = 1000
	xValidate, YValidate, yValidate = loadBatch("test_batch")
	xValidate= xValidate[:,0:numVal]
	YValidate= YValidate[:,0:numVal]
	yValidate= yValidate[0:numVal]

	return xTrain, YTrain, yTrain, xValidate, YValidate, yValidate

def getInitData(X,Y, Xavier=False):
	var = 0.01
	if Xavier:
		var = 1/X.shape[0]
	W = np.matrix([[np.random.normal(0,var) for d in range(X.shape[0])] for K in range(Y.shape[0])])
	b = np.matrix([[np.random.normal(0,var)] for K in range(Y.shape[0])])
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
	L2 = np.sum(np.power(W,2))
	return (lCross/X.shape[1] + lamda*L2).item(0)

def computeSVMCost(X, y, W, b, lamda):
	total = 0.0
	for i in range(X.shape[1]):
		scores = W*X[:,i:i+1] + b
		correctScore = scores[y[i]]
		for j in range(scores.shape[0]):
			if j == y[i]:
				continue
			total += max(0, scores[j] - correctScore + 1)
	L2 = np.sum(np.power(W,2))
	return (total/X.shape[1] + lamda*L2).item(0)

def computeSVMGradients(X, y, W, b, lamda):
	gradB = np.zeros((W.shape[0], 1))
	gradW = np.zeros(W.shape)

	for i in range(X.shape[1]):
		scores = W*X[:,i:i+1] + b
		correctScore = scores[y[i]]
		for j in range(scores.shape[0]):
			if j == y[i]:
				continue
			margin = scores[j] - correctScore + 1
			if margin > 0:
				gradW[y[i]] -= X[:,i]
				gradW[j] += X[:,i]
				gradB[j] += 1.0
				gradB[y[i]] -= 1.0
	gradW /= X.shape[1]
	gradB /= X.shape[1]
	gradW += 2*lamda*W
	return gradW, gradB

def computeAccuracy(X, y, W, b):
	P = evaluateClassifier(X,W,b)
	corrects = 0.0
	for i in range(len(y)):
		if y[i] == np.argmax(P[:,i]):
			corrects+=1
	return corrects/len(y)

def computeGradients(X, Y, W, b, lamda):
	P = evaluateClassifier(X,W,b)
	g = P-Y
	gradB = np.zeros((Y.shape[0], 1))
	gradW = np.zeros(W.shape)
	for i in range(g.shape[1]):
		gradB += g[:,i]
		gradW += g[:,i]*X[:,i].T
	gradB /= X.shape[1]
	gradW /= X.shape[1]
	gradW += 2*lamda*W
	return gradW, gradB

def computeGradsNum(X, Y, W, b, lamda, h):
	no = W.shape[0]
	d = X.shape[0]

	gradW = np.zeros(W.shape)
	gradB = np.zeros((no, 1))

	c = computeSVMCost(X, Y, W, b, lamda)

	for i in range(b.shape[0]):
		bTry = b.copy()
		bTry[i] += h
		c2 = computeSVMCost(X, Y, W, bTry, lamda)
		gradB[i] = (c2-c)/h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			WTry = W.copy()
			WTry[i,j] += h
			c2 = computeSVMCost(X, Y, WTry, b, lamda)
			gradW[i,j] = (c2-c)/h

	return gradW, gradB

def updateNetwork(X, Y, GDparams, W, b, lamda):
	gradW, gradB = computeGradients(X, Y, W, b, lamda)
	W -= GDparams[1]*gradW
	b -= GDparams[1]*gradB

def updateNetworkSVM(X, y, GDparams, W, b, lamda):
	gradW, gradB = computeSVMGradients(X, y, W, b, lamda)
	W -= GDparams[1]*gradW
	b -= GDparams[1]*gradB

def miniBatchGD(X, Y, y, GDparams, W, b, lamda, XV, YV, yV, earlyStop=False):

	costTrain = [0.0]*GDparams[2]
	accTrain = [0.0]*GDparams[2]
	costVal = [0.0]*GDparams[2]
	accVal = [0.0]*GDparams[2]

	stoppedAt = 0
	for epoch in range(GDparams[2]):
		print(epoch+1)
		stoppedAt = epoch + 1
		for i in range(1, math.floor(X.shape[1]/GDparams[0])):
			start = (i-1)*GDparams[0]
			end = i*GDparams[0]
			XBatch = X[:,start:end]
			YBatch = Y[:,start:end]
			updateNetwork(XBatch, YBatch, GDparams, W, b, lamda)
		costTrain[epoch] = computeCost(X, Y, W, b, lamda)
		accTrain[epoch] = computeAccuracy(X, y, W, b)
		costVal[epoch] = computeCost(XV, YV, W, b, lamda)
		accVal[epoch] = computeAccuracy(XV, yV, W, b)
		if earlyStop and epoch > 0 and costVal[epoch - 1] < costVal[epoch]:
			break
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

def miniBatchGDSVM(X, Y, y, GDparams, W, b, lamda, XV, YV, yV, earlyStop=False):

	costTrain = [0.0]*GDparams[2]
	accTrain = [0.0]*GDparams[2]
	costVal = [0.0]*GDparams[2]
	accVal = [0.0]*GDparams[2]

	stoppedAt = 0
	for epoch in range(GDparams[2]):
		print(epoch+1)
		stoppedAt = epoch + 1
		for i in range(1, math.floor(X.shape[1]/GDparams[0])):
			start = (i-1)*GDparams[0]
			end = i*GDparams[0]
			XBatch = X[:,start:end]
			yBatch = y[start:end]
			updateNetworkSVM(XBatch, yBatch, GDparams, W, b, lamda)
		costTrain[epoch] = computeSVMCost(X, y, W, b, lamda)
		accTrain[epoch] = computeAccuracy(X, y, W, b)
		costVal[epoch] = computeSVMCost(XV, yV, W, b, lamda)
		accVal[epoch] = computeAccuracy(XV, yV, W, b)
		if earlyStop and epoch > 0 and costVal[epoch - 1] < costVal[epoch]:
			break
	labels = []
	labels.append(plt.plot(costTrain[0:stoppedAt], label="Training")[0])
	labels.append(plt.plot(costVal[0:stoppedAt], label="Validation")[0])
	plt.legend(handles=labels)
	plt.title("SVM cost")
	plt.ylabel("Cost")
	plt.xlabel("Epoch")
	plt.show()
	labels = []
	labels.append(plt.plot(accTrain[0:stoppedAt], label="Training")[0])
	labels.append(plt.plot(accVal[0:stoppedAt], label="Validation")[0])
	plt.legend(handles=labels)
	plt.title("SVM Accuracy")
	plt.ylabel("Accuracy")
	plt.xlabel("Epoch")
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

def checkGradTest():
	X, Y, y = loadBatch("data_batch_1")
	n = 10
	X = X[:,0:n]
	Y = Y[:,0:n]
	y = y[0:n]
	W, b = getInitData(X,Y)
	lamda = 0.0
	analW, analB = computeSVMGradients(X, y, W, b, lamda)
	numW, numB = computeGradsNum(X, y, W, b, lamda, 1e-06)
	wError = np.max(abs(analW - numW) / np.clip(abs(analW) + abs(numW), a_min=1e-06, a_max=9999))
	bError = np.max(abs(analB - numB) / np.clip(abs(analB) + abs(numB), a_min=1e-06, a_max=9999))
	print("W = " + str(wError))
	print("b = " + str(bError))

	print("W = " + str(np.max(abs(analW - numW))))
	print("b = " + str(np.max(abs(analB - numB))))

#checkGradTest()

def test():
	lamda = 0.0
	GDparams = [100, 0.005, 20]
	X, Y, y, XValidate, YValidate, yValidate = getAllData()
	W, b = getInitData(X,Y, Xavier=True)
	miniBatchGDSVM(X, Y, y, GDparams, W, b, lamda, XValidate, YValidate, yValidate, earlyStop=False)

test()

#printW(W)

#imgShow(file[b"data"][np.random.randint(0, len(file[b"data"]))].T)