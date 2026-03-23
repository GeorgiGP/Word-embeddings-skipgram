#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################
###
### Домашно задание 2 -- помощни функции
###
#############################################################################

import nltk
from nltk.corpus import PlaintextCorpusReader
import sys
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD

#############################################################
###  Визуализация на прогреса
#############################################################
class progressBar:
    def __init__(self ,barWidth = 50):
        self.barWidth = barWidth
        self.period = None
    def start(self, count):
        self.item=0
        self.period = int(count / self.barWidth)
        sys.stdout.write("["+(" " * self.barWidth)+"]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth+1))
    def tick(self):
        if self.item>0 and self.item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
        self.item += 1
    def stop(self):
        sys.stdout.write("]\n")

#############################################################
###  Извличане на речник --
###  връща списъка от по-често срещаните думи,
###  индексите им и честотите им
#############################################################
def extractDictionary(corpus, limit=20000):
    pb = progressBar()
    pb.start(len(corpus))
    dictionary = {}
    for doc in corpus:
        pb.tick()
        for w in doc:
            if w not in dictionary: dictionary[w] = 0
            dictionary[w] += 1
    L = sorted([(w,dictionary[w]) for w in dictionary], key = lambda x: x[1] , reverse=True)
    if limit > len(L): limit = len(L)
    words = [ w for w,_ in L[:limit] ]
    word2ind = { w:i for i,w in enumerate(words)}
    freqs = [ f for _,f in L[:limit] ]
    pb.stop()
    return words, word2ind, freqs


#############################################################
###  Редуциране на измеренията на матрица чрез SVD
#############################################################
def SVD_k_dim(X, k=100, n_iters = 10):
    print("Running Truncated SVD over %i words..." % (X.shape[0]))
    svd = TruncatedSVD(n_components=k, n_iter=n_iters)
    svd.fit(X)
    X_reduced = svd.transform(X)
    print("Done.")
    return X_reduced

#############################################################
###  Двумерна визуализация на влаганията на списък от думи
#############################################################
def plot_embeddings(M, word2ind, words, filename=None):
    xs = M[:,0]
    ys = M[:,1]
    for w in words:
        i=word2ind[w]
        plt.scatter(xs[i],ys[i], marker='x', color= 'red')
        plt.text(xs[i]+0.001, ys[i]+0.001, w)
    if filename: plt.savefig(filename)
    plt.show()

#############################################################
###  Извличане на данните от корпуса --
###  връща списъка от двойки (целева дума, контекстна дума)
#############################################################
def extractWordContextPairs(corpus, windowSize, word2ind):
    pb = progressBar()
    pb.start(len(corpus))
    data = []
    for doc in corpus:
        pb.tick()
        for wi in range(len(doc)):
            if doc[wi] not in word2ind: continue
            w = word2ind[doc[wi]]
            for k in range(1,windowSize+1):
                if wi-k>=0 and doc[wi-k] in word2ind:
                    c = word2ind[doc[wi-k]]
                    data.append((w,c))
                if wi+k<len(doc) and doc[wi+k] in word2ind:
                    c = word2ind[doc[wi+k]]
                    data.append((w,c))
    pb.stop()
    return data

#############################################################
