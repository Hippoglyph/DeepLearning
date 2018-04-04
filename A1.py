import numpy as np
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
	return (lCross/X.shape[0] + lamda*L2).item(0)

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


lamda = 0.01
X, Y, y = loadBatch("data_batch_1")
W, b = getInitData(X,Y)

#print(computeCost(X,Y,W,b, lamda))
#print(computeAccuracy(X,y,W,b))
print(computeGradients(X,Y,W,lamda))
#imgShow(file[b"data"][np.random.randint(0, len(file[b"data"]))].T)