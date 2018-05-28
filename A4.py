import numpy as np

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


def fowardPass(RNN, hp, x):
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
		p, hp = fowardPass(RNN, hp, X[:,t:t+1])
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

def getInput(X,Y):
	xIN = np.zeros((K,len(X)))
	yIN = np.zeros((K,len(Y)))
	for i in range(len(X)):
		xIN[charToInd[X[i]],i] = 1
		yIN[charToInd[Y[i]],i] = 1
	return xIN, yIN

def test():
	RNN = initRNN()
	xChars = bookData[0:seqLength]
	yChars = bookData[1:seqLength+1]

	xTrain, yTrain = getInput(xChars, yChars)
	#hp = np.zeros((m,1))
	calculateGradient(RNN,xTrain,yTrain)
		

m = 100
eta = .1
seqLength = 25
sig = .01

K, charToInd, indToChar, bookData = getData()



test()