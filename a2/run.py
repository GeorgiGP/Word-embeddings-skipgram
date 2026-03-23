#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################
###
### Домашно задание 2  -- програма за извикване на обучението
###
#############################################################################

import nltk
from nltk.corpus import PlaintextCorpusReader
import numpy as np
import sys

import grads
import utils
import w2v_sgd
import sampling

#############################################################
#######   Зареждане на корпуса
#############################################################
startToken = '<START>'
endToken = '<END>'

import nltk
nltk.download('punkt_tab')

corpus_root = '../../JOURNALISM.BG/C-MassMedia'
myCorpus = PlaintextCorpusReader(corpus_root, '.*\.txt')

corpus = [ [startToken] + [w.lower() for w in sent] + [endToken] for sent in myCorpus.sents()]

windowSize = 3
negativesCount = 5
embDim = 50

words, word2ind, freqs = utils.extractDictionary(corpus, limit=20000)
data = utils.extractWordContextPairs(corpus, windowSize, word2ind)

del corpus

U0 = (np.random.rand(len(words), embDim) - 0.5) / embDim
V0 = (np.random.rand(len(words), embDim) - 0.5) / embDim
W0 = (np.random.rand(embDim, embDim) - 0.5) / embDim

seq = sampling.createSamplingSequence(freqs)
q_noise = sampling.noiseDistribution(freqs,negativesCount)

contextFunction = lambda c: sampling.sampleContext(c, seq, negativesCount)

if len(sys.argv)>1 and sys.argv[1] == 'cumulative':
    lossAndGradientFunction = grads.lossAndGradientCumulative
else:
	lossAndGradientFunction = grads.lossAndGradientBatched
U,V,W = w2v_sgd.stochasticGradientDescend(data,np.copy(U0),np.copy(V0),np.copy(W0),contextFunction,lossAndGradientFunction,q_noise)

np.save('w2v-U',U)
np.save('w2v-V',V)
np.save('w2v-W',W)

E = np.concatenate([np.dot(V,W),U],axis=1)

E_reduced =utils.SVD_k_dim(E,k=2)
E_normalized_2d = E_reduced /np.linalg.norm(E_reduced, axis=1)[:, np.newaxis]

sampleWords = 'януари октомври седмица година медии пазар стоки бизнес фирма бюджет петрол нефт'.split()

utils.plot_embeddings(E_normalized_2d, word2ind, sampleWords, 'embeddings')


