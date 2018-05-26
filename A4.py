import numpy as np

class RNNClass:
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

	return K, charToInd, indToChar

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

def predict(RNN, hp = [], xC = "\n", n=140):
	if hp == []:
		hp = np.zeros((m,1))
	x = charToInd[xC]
	pred = xC
	for j in range(n):
		p, hp = fowardPass(RNN, hp, x)
		x = sample(p)
		pred += indToChar[x]
	print(pred)


def fowardPass(RNN, hp, x):
	xt = np.zeros((K,1))
	xt[x] = 1
	a = np.matmul(RNN.W,hp) + np.matmul(RNN.U,xt) + RNN.b
	h = np.tanh(a)
	o = np.matmul(RNN.V,h) + RNN.c
	p = softmax(o)
	return p, h

def test():
	RNN = initRNN()
	predict(RNN)
	

K, charToInd, indToChar = getData()

m = 100
eta = .1
seqLength = 25
sig = .01

test()