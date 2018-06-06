import numpy as np
import copy
import sys

class RNNClass:
	pass

class RNNGradClass:
	pass

def getData():
	with open("goblet_book.txt", "r",encoding='utf-8') as f:
		bookData = f.read()
	bookChars = list(set(bookData))
	K = len(bookChars)
	charToInd = {}
	indToChar = {}
	for i in range(K):
		charToInd[bookChars[i]] = i
		indToChar[i] = bookChars[i]

	return K, charToInd, indToChar, bookData

def initRNN():
	RNN = RNNClass()
	RNN.b = np.zeros((m,1))
	RNN.c = np.zeros((K,1))
	RNN.U = np.random.rand(m,K)*sig
	RNN.W = np.random.rand(m,m)*sig
	RNN.V = np.random.rand(K,m)*sig
	return RNN

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sample(p):
	t = np.random.random()
	summ = .0
	for i in range(K):
		summ += p[i]
		if summ > t:
			return i
	return K-1

def predict(RNN, n=140):
	hp = np.zeros((m,1))
	x = np.zeros((K,1))
	prevXi = 0
	x[prevXi] = 1
	
	pred = ""
	for j in range(n):
		p, hp = fowardPass(RNN, hp, x)
		xi = sample(p)
		pred += indToChar[xi]
		x[prevXi] = 0
		x[xi] = 1
	print(pred)

def computeLoss(RNN, X, Y):
	P, _ = feedForward(RNN, X)
	error = .0
	for t in range(P.shape[1]):
		error -= np.log(np.dot(Y[:,t:t+1].T,P[:,t:t+1])).item(0)
	return error

def forwardPass(RNN, hp, x):
	a = np.matmul(RNN.W,hp) + np.matmul(RNN.U,x) + RNN.b
	h = np.tanh(a)
	o = np.matmul(RNN.V,h) + RNN.c
	p = softmax(o)
	return p, h

def feedForward(RNN, X):
	hp = np.zeros((m,1))
	hL = np.zeros((m,seqLength+1))
	pL = np.zeros((K,seqLength))
	for t in range(X.shape[1]):
		p, hp = forwardPass(RNN, hp, X[:,t:t+1])
		pL[:,t:t+1] = p
		hL[:,t+1:t+2] = hp
	return pL, hL

def calculateGradient(RNN,X,Y):

	RNNGrad = RNNGradClass()
	RNNGrad.b = np.zeros((m,1))
	RNNGrad.c = np.zeros((K,1))
	RNNGrad.U = np.zeros((m,K))
	RNNGrad.W = np.zeros((m,m))
	RNNGrad.V = np.zeros((K,m))
	RNNGrad.A = np.zeros((m,1))

	p,h = feedForward(RNN, X)

	for t in reversed(range(X.shape[1])):
		RNNGrad.o = p[:,t:t+1] - Y[:,t:t+1]
		RNNGrad.V += np.matmul(RNNGrad.o, h[:,t+1:t+2].T)
		RNNGrad.c += RNNGrad.o
		RNNGrad.H = np.matmul(RNN.V.T, RNNGrad.o) + np.matmul(RNN.W.T, RNNGrad.A)
		RNNGrad.A = np.multiply(RNNGrad.H, 1- np.power(h[:,t+1:t+2],2))
		RNNGrad.W += np.matmul(RNNGrad.A, h[:,t:t+1].T)
		RNNGrad.b += RNNGrad.A
		RNNGrad.U += np.matmul(RNNGrad.A, X[:,t:t+1].T)

	return RNNGrad

def cumputeGradsNum(RNN, X, Y, h = 1e-5):

	RNNGrad = RNNGradClass()
	RNNGrad.b = np.zeros((m,1))
	RNNGrad.c = np.zeros((K,1))
	RNNGrad.U = np.zeros((m,K))
	RNNGrad.W = np.zeros((m,m))
	RNNGrad.V = np.zeros((K,m))

	progIter = 0
	progToral = m + K + m*K + m*m + K*m

	for i in range(RNN.b.shape[0]):
		RNNTry = copy.deepcopy(RNN)
		RNNTry.b[i] -= h
		l1 = computeLoss(RNNTry, X, Y)
		RNNTry.b[i] += 2*h
		l2 = computeLoss(RNNTry, X, Y)
		RNNGrad.b[i] = (l2-l1)/(2*h)
		progIter += 1
		progressPrint(progIter, progToral)

	for i in range(RNN.c.shape[0]):
		RNNTry = copy.deepcopy(RNN)
		RNNTry.c[i] -= h
		l1 = computeLoss(RNNTry, X, Y)
		RNNTry.c[i] += 2*h
		l2 = computeLoss(RNNTry, X, Y)
		RNNGrad.c[i] = (l2-l1)/(2*h)
		progIter += 1
		progressPrint(progIter, progToral)

	for i in range(RNN.U.shape[0]):
		for j in range(RNN.U.shape[1]):
			RNNTry = copy.deepcopy(RNN)
			RNNTry.U[i,j] -= h
			l1 = computeLoss(RNNTry, X, Y)
			RNNTry.U[i,j] += 2*h
			l2 = computeLoss(RNNTry, X, Y)
			RNNGrad.U[i,j] = (l2-l1)/(2*h)
			progIter += 1
			progressPrint(progIter, progToral)

	for i in range(RNN.W.shape[0]):
		for j in range(RNN.W.shape[1]):
			RNNTry = copy.deepcopy(RNN)
			RNNTry.W[i,j] -= h
			l1 = computeLoss(RNNTry, X, Y)
			RNNTry.W[i,j] += 2*h
			l2 = computeLoss(RNNTry, X, Y)
			RNNGrad.W[i,j] = (l2-l1)/(2*h)
			progIter += 1
			progressPrint(progIter, progToral)

	for i in range(RNN.V.shape[0]):
		for j in range(RNN.V.shape[1]):
			RNNTry = copy.deepcopy(RNN)
			RNNTry.V[i,j] -= h
			l1 = computeLoss(RNNTry, X, Y)
			RNNTry.V[i,j] += 2*h
			l2 = computeLoss(RNNTry, X, Y)
			RNNGrad.V[i,j] = (l2-l1)/(2*h)
			progIter += 1
			progressPrint(progIter, progToral)

	sys.stdout.write('\r'+"100%  ")
	sys.stdout.flush()
	print("")

	return RNNGrad

def checkGradTest():
	RNN = initRNN()
	xChars = bookData[0:seqLength]
	yChars = bookData[1:seqLength+1]
	X, Y = getInput(xChars, yChars)

	analGrads = calculateGradient(RNN,X,Y)
	numGrads = cumputeGradsNum(RNN,X,Y)

	bError = np.max(abs(analGrads.b - numGrads.b) / np.clip(abs(analGrads.b) + abs(numGrads.b), a_min=1e-06, a_max=9999))
	cError = np.max(abs(analGrads.c - numGrads.c) / np.clip(abs(analGrads.c) + abs(numGrads.c), a_min=1e-06, a_max=9999))
	UError = np.max(abs(analGrads.U - numGrads.U) / np.clip(abs(analGrads.U) + abs(numGrads.U), a_min=1e-06, a_max=9999))
	WError = np.max(abs(analGrads.W - numGrads.W) / np.clip(abs(analGrads.W) + abs(numGrads.W), a_min=1e-06, a_max=9999))
	VError = np.max(abs(analGrads.V - numGrads.V) / np.clip(abs(analGrads.V) + abs(numGrads.V), a_min=1e-06, a_max=9999))

	print("c = " + str(cError))
	print("v = " + str(VError))
	print("b = " + str(bError))
	print("U = " + str(UError))
	print("W = " + str(WError))

def getInput(X,Y):
	xIN = np.zeros((K,len(X)))
	yIN = np.zeros((K,len(Y)))
	for i in range(len(X)):
		xIN[charToInd[X[i]],i] = 1
		yIN[charToInd[Y[i]],i] = 1
	return xIN, yIN

def progressPrint(nominator, denominator):
	denominator = float(denominator)*100
	nominator = float(nominator)*100
	if(nominator % (denominator/100) == 0):
		number = str(round((nominator/denominator)*100, 0))
		sys.stdout.write('\r'+ number + "%")
		sys.stdout.flush()

def test():
	RNN = initRNN()
	xChars = bookData[0:seqLength]
	yChars = bookData[1:seqLength+1]

	xTrain, yTrain = getInput(xChars, yChars)
	print(computeLoss(RNN,xTrain,yTrain))
		

m = 100
eta = .1
seqLength = 25
sig = .01

K, charToInd, indToChar, bookData = getData()



#test()
checkGradTest()