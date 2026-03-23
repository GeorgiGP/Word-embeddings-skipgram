#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################
###
### Домашно задание 2  -- тестове
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
import pickle


if len(sys.argv)>1:
    if sys.argv[1] == '3':
        try:
            with open ('test3', 'rb') as fp: freqs = pickle.load(fp)
            seq = sampling.createSamplingSequence(freqs)
            assert set(seq) == set([*range(20000)]), "Елементите на seq следва да са числата от 0 до 19999"
            assert len(seq) == 1233319, "Броят на елементите на seq не съвпада с очаквания"
            print("Функцията createSamplingSequence премина теста.")
        except Exception as exception:
            print("Грешка при тестването на createSamplingSequence -- "+str(exception))
        try:
            q_noise = sampling.noiseDistribution(freqs,10)
            assert np.abs(np.sum(np.exp(q_noise)/10) - 1.0) < 1e-7, "Сумата от вероатностите за контекстните думи следва да е 1.0"
            assert np.max(q_noise) < 0. , "Максималната вероятност за контекстна дума следва да е по-малка от 1"
            assert np.exp(np.min(q_noise))/10 > 7e-6 , "Минималната вероятност за контекстна дума следва да е по-голяма от 18/10038361 ** 0.75 "
            print("Функцията noiseDistribution премина теста.")
        except Exception as exception:
            print("Грешка при тестването на noiseDistribution -- "+str(exception))
    elif sys.argv[1] == '4':
        try:
            [u_w,Vt,W,q,J, du_w, dVt, dW] = np.load('test4.npy',allow_pickle=True)
            J1, du_w1, dVt1, dW1 = grads.lossAndGradient(u_w, Vt, W, q)
            assert du_w.shape == du_w1.shape, "Формата на du_w не съвпада с очакваната"
            assert dVt.shape == dVt1.shape, "Формата на dVt не съвпада с очакваната"
            assert dW.shape == dW1.shape, "Формата на dW не съвпада с очакваната"
            assert np.max(np.abs(J-J1))<1e-7, "Стойноста на J не съвпадат с очакваната"
            assert np.max(np.abs(du_w-du_w1))<1e-7, "Стойностите на du_w не съвпадат с очакваните"
            assert np.max(np.abs(dVt-dVt1))<1e-7, "Стойностите на dVt не съвпадат с очакваните"
            assert np.max(np.abs(dW-dW1))<1e-7, "Стойностите на dW не съвпадат с очакваните"
            print("Функцията lossAndGradient премина теста.")
        except Exception as exception:
            print("Грешка при тестването на lossAndGradient -- "+str(exception))
    elif sys.argv[1] == '5':
        try:
            [u_w,Vt,W,q,J,du_w,dVt,dW] = np.load('test5.npy',allow_pickle=True)
            J1, du_w1, dVt1, dW1 = grads.lossAndGradientBatched(u_w, Vt, W, q)
            assert du_w.shape == du_w1.shape, "Формата на du_w не съвпада с очакваната"
            assert dVt.shape == dVt1.shape, "Формата на dVt не съвпада с очакваната"
            assert dW.shape == dW1.shape, "Формата на dW не съвпада с очакваната"
            assert np.max(np.abs(J-J1))<1e-7, "Стойноста на J не съвпадат с очакваната"
            assert np.max(np.abs(du_w-du_w1))<1e-7, "Стойностите на du_w не съвпадат с очакваните"
            assert np.max(np.abs(dVt-dVt1))<1e-7, "Стойностите на dVt не съвпадат с очакваните"
            assert np.max(np.abs(dW-dW1))<1e-7, "Стойностите на dW не съвпадат с очакваните"
            print("Функцията lossAndGradientBatched премина теста.")
        except Exception as exception:
            print("Грешка при тестването на lossAndGradientBatched -- "+str(exception))
    elif sys.argv[1] == '6':
        try:
            [data,U0,V0,W0,U,V,W] = np.load('test6.npy',allow_pickle=True)
            contextFunction = lambda c: [c,4543,6534,12345,9321,1234]
            q_noise = np.log(np.ones(20000)*5.0/20000.0)
            U1,V1,W1 = w2v_sgd.stochasticGradientDescend(data,np.copy(U0),np.copy(V0),np.copy(W0),contextFunction,grads.lossAndGradientCumulative,q_noise, batchSize=100000)
            assert U.shape == U1.shape, "Формата на U не съвпада с очакваната"
            assert V.shape == V1.shape, "Формата на V не съвпада с очакваната"
            assert W.shape == W1.shape, "Формата на W не съвпада с очакваната"
            assert np.max(np.abs(U-U1))<1e-7, "Стойностите на U не съвпадат с очакваните"
            assert np.max(np.abs(V-V1))<1e-7, "Стойностите на V не съвпадат с очакваните"
            assert np.max(np.abs(W-W1))<1e-7, "Стойностите на W не съвпадат с очакваните"
            print("Функцията stochasticGradientDescend премина теста.")
        except Exception as exception:
            print("Грешка при тестването на stochasticGradientDescend -- "+str(exception))
