import numpy as np
import copy
import sys

class RNNClass:
	pass

class RNNGradClass:
	pass

class AdaGradClass:
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
	print("Should never be here")
	return K-1

def predict(RNN, hp, xChar, n=200):
	nhp = copy.deepcopy(hp)
	x = np.zeros((K,1))

	prevXi = charToInd[xChar]
	x[prevXi] = 1
	
	pred = ""
	for j in range(n):
		p, nhp = forwardPass(RNN, nhp, x)
		xi = sample(p)
		pred += indToChar[xi]
		x[prevXi] = 0
		x[xi] = 1
		prexXi = xi
	print(pred)

def computeLoss(RNN, X, Y, hp):
	P, h = feedForward(RNN, X, hp)
	loss = .0
	for t in range(P.shape[1]):
		loss -= np.log(np.dot(Y[:,t:t+1].T,P[:,t:t+1])).item(0)
	return loss, P, h

def forwardPass(RNN, hp, x):
	a = np.matmul(RNN.W,hp) + np.matmul(RNN.U,x) + RNN.b
	h = np.tanh(a)
	o = np.matmul(RNN.V,h) + RNN.c
	p = softmax(o)
	return p, h

def feedForward(RNN, X, hp):
	hL = np.zeros((m,seqLength+1))
	hL[:,0:1] = hp
	pL = np.zeros((K,seqLength))
	for t in range(X.shape[1]):
		p, nhp = forwardPass(RNN, hL[:,t:t+1], X[:,t:t+1])
		pL[:,t:t+1] = p
		hL[:,t+1:t+2] = nhp
	return pL, hL

def calculateGradient(RNN,X,Y, hp):

	RNNGrad = RNNGradClass()
	RNNGrad.b = np.zeros((m,1))
	RNNGrad.c = np.zeros((K,1))
	RNNGrad.U = np.zeros((m,K))
	RNNGrad.W = np.zeros((m,m))
	RNNGrad.V = np.zeros((K,m))
	RNNGrad.A = np.zeros((m,1))

	loss,p,h = computeLoss(RNN, X, Y, hp)

	for t in reversed(range(X.shape[1])):
		RNNGrad.o = p[:,t:t+1] - Y[:,t:t+1]
		RNNGrad.V += np.matmul(RNNGrad.o, h[:,t+1:t+2].T)
		RNNGrad.c += RNNGrad.o
		RNNGrad.H = np.matmul(RNN.V.T, RNNGrad.o) + np.matmul(RNN.W.T, RNNGrad.A)
		RNNGrad.A = np.multiply(RNNGrad.H, 1- np.power(h[:,t+1:t+2],2))
		RNNGrad.W += np.matmul(RNNGrad.A, h[:,t:t+1].T)
		RNNGrad.b += RNNGrad.A
		RNNGrad.U += np.matmul(RNNGrad.A, X[:,t:t+1].T)

	return RNNGrad, h[:,-2:-1], loss

def cumputeGradsNum(RNN, X, Y, hp, h = 1e-5):

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
		l1,_,_ = computeLoss(RNNTry, X, Y, hp)
		RNNTry.b[i] += 2*h
		l2,_,_ = computeLoss(RNNTry, X, Y, hp)
		RNNGrad.b[i] = (l2-l1)/(2*h)
		progIter += 1
		progressPrint(progIter, progToral)

	for i in range(RNN.c.shape[0]):
		RNNTry = copy.deepcopy(RNN)
		RNNTry.c[i] -= h
		l1,_,_ = computeLoss(RNNTry, X, Y, hp)
		RNNTry.c[i] += 2*h
		l2,_,_ = computeLoss(RNNTry, X, Y, hp)
		RNNGrad.c[i] = (l2-l1)/(2*h)
		progIter += 1
		progressPrint(progIter, progToral)

	for i in range(RNN.U.shape[0]):
		for j in range(RNN.U.shape[1]):
			RNNTry = copy.deepcopy(RNN)
			RNNTry.U[i,j] -= h
			l1,_,_ = computeLoss(RNNTry, X, Y, hp)
			RNNTry.U[i,j] += 2*h
			l2,_,_ = computeLoss(RNNTry, X, Y, hp)
			RNNGrad.U[i,j] = (l2-l1)/(2*h)
			progIter += 1
			progressPrint(progIter, progToral)

	for i in range(RNN.W.shape[0]):
		for j in range(RNN.W.shape[1]):
			RNNTry = copy.deepcopy(RNN)
			RNNTry.W[i,j] -= h
			l1,_,_ = computeLoss(RNNTry, X, Y, hp)
			RNNTry.W[i,j] += 2*h
			l2,_,_ = computeLoss(RNNTry, X, Y, hp)
			RNNGrad.W[i,j] = (l2-l1)/(2*h)
			progIter += 1
			progressPrint(progIter, progToral)

	for i in range(RNN.V.shape[0]):
		for j in range(RNN.V.shape[1]):
			RNNTry = copy.deepcopy(RNN)
			RNNTry.V[i,j] -= h
			l1,_,_ = computeLoss(RNNTry, X, Y, hp)
			RNNTry.V[i,j] += 2*h
			l2,_,_ = computeLoss(RNNTry, X, Y, hp)
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

	hp = np.zeros((m,1))
	analGrads,_,_ = calculateGradient(RNN,X,Y, hp)
	numGrads = cumputeGradsNum(RNN,X,Y,hp)

	bError = np.max(abs(analGrads.b - numGrads.b) / np.clip(abs(analGrads.b) + abs(numGrads.b), a_min=1e-06, a_max=9999))
	cError = np.max(abs(analGrads.c - numGrads.c) / np.clip(abs(analGrads.c) + abs(numGrads.c), a_min=1e-06, a_max=9999))
	UError = np.max(abs(analGrads.U - numGrads.U) / np.clip(abs(analGrads.U) + abs(numGrads.U), a_min=1e-06, a_max=9999))
	WError = np.max(abs(analGrads.W - numGrads.W) / np.clip(abs(analGrads.W) + abs(numGrads.W), a_min=1e-06, a_max=9999))
	VError = np.max(abs(analGrads.V - numGrads.V) / np.clip(abs(analGrads.V) + abs(numGrads.V), a_min=1e-06, a_max=9999))

	print("c = " + str(cError))
	print("V = " + str(VError))
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

def clipGrads(RNNGrad):
	RNNGrad.b = np.clip(RNNGrad.b, -5, 5)
	RNNGrad.c = np.clip(RNNGrad.c, -5, 5)
	RNNGrad.U = np.clip(RNNGrad.U, -5, 5)
	RNNGrad.W = np.clip(RNNGrad.W, -5, 5)
	RNNGrad.V = np.clip(RNNGrad.V, -5, 5)

def progressPrint(nominator, denominator):
	denominator = float(denominator)*100
	nominator = float(nominator)*100
	if(nominator % (denominator/100) == 0):
		number = str(round((nominator/denominator)*100, 0))
		sys.stdout.write('\r'+ number + "%")
		sys.stdout.flush()

def fit(RNN, AdaGrad, e, hp):
	xChars = bookData[e:e+seqLength]
	yChars = bookData[e+1:e+seqLength+1]
	xTrain, yTrain = getInput(xChars, yChars)
	RNNGrads, h, loss = calculateGradient(RNN, xTrain, yTrain, hp)
	clipGrads(RNNGrads)

	AdaGrad.b += np.power(RNNGrads.b,2)
	AdaGrad.c += np.power(RNNGrads.c,2)
	AdaGrad.U += np.power(RNNGrads.U,2)
	AdaGrad.W += np.power(RNNGrads.W,2)
	AdaGrad.V += np.power(RNNGrads.V,2)

	RNN.b -= AdaGrad.n*np.divide(RNNGrads.b, np.sqrt(AdaGrad.b + AdaGrad.eps))
	RNN.c -= AdaGrad.n*np.divide(RNNGrads.c, np.sqrt(AdaGrad.c + AdaGrad.eps))
	RNN.U -= AdaGrad.n*np.divide(RNNGrads.U, np.sqrt(AdaGrad.U + AdaGrad.eps))
	RNN.W -= AdaGrad.n*np.divide(RNNGrads.W, np.sqrt(AdaGrad.W + AdaGrad.eps))
	RNN.V -= AdaGrad.n*np.divide(RNNGrads.V, np.sqrt(AdaGrad.V + AdaGrad.eps))

	return h, loss

def initAdagrad():
	AdaGrad = AdaGradClass()
	AdaGrad.b = np.zeros((m,1))
	AdaGrad.c = np.zeros((K,1))
	AdaGrad.U = np.zeros((m,K))
	AdaGrad.W = np.zeros((m,m))
	AdaGrad.V = np.zeros((K,m))

	AdaGrad.n = 1e-2
	AdaGrad.eps = 1e-5

	return AdaGrad

def train(RNN, numEpoch):
	AdaGrad = initAdagrad()
	iterN = 0
	smoothLoss = 0.0
	progVar = -1
	printLimit = 10000

	for epoch in range(numEpoch):
		e = 0
		hp = np.zeros((m,1))
		while(e + seqLength + 1 < len(bookData)):
			hp, loss = fit(RNN, AdaGrad, e, hp)
			smoothLoss = loss if smoothLoss == 0.0 else .999*smoothLoss + .001 * loss
			e += seqLength
			if iterN % printLimit == 0:
				sys.stdout.write("\r     ")
				sys.stdout.flush()
				print("")
				progVar+=1
				print("------")
				print("Epoch = " + str(epoch+1) +", Iter = " + str(iterN) + ", Smooth Loss = " + str(smoothLoss))
				predict(RNN, hp, bookData[e-1])
			progressPrint(iterN - printLimit* progVar, printLimit)
			iterN += 1

def test():
	RNN = initRNN()

	train(RNN, 10)

m = 100
eta = .1
seqLength = 25
sig = .01

K, charToInd, indToChar, bookData = getData()


test()
#checkGradTest()