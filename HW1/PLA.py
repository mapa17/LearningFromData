# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:53:05 2015

@author: Manuel Pasieka , manuel.pasieka@csf.ac.at

Homework Assignment from https://work.caltech.edu/telecourse.html

Simple Perceptron Learning Model Algorithm
"""

import sys
import numpy as np
import random
import matplotlib.pyplot as plt


def getRandomLine(seed=None):
    """
    Generates Point of Reference A and a direction g

    >>> getRandomLine(seed=1)
    (array([[ 0.73127151, -0.69486747]]), array([[-0.52754924,  0.48986195]]))
    >>> A, g = getRandomLine(seed=1)
    >>> A
    array([[ 0.73127151, -0.69486747]])
    """
    np.random.seed(seed)
    A = np.ones((1,2)) - 2 * np.random.random((1,2))
    g = np.ones((1,2)) - 2 * np.random.random((1,2))
    return A, g


def getRandomPoints(N, seed=None):
    """
    Get N 2D vectors ranging from (-1, 1)
    Vectors are a np.array of size (N, 2)

    >>> getRandomPoints(10, seed=13)
    array([[ 0.48198302, -0.37051599],
            [-0.36816384, -0.69867232],
            [ 0.62855165,  0.53888278],
            [ 0.70568016,  0.54967413],
            [-0.4680472 ,  0.73957395],
            [-0.0626295 ,  0.57218494],
            [ 0.41068649,  0.13683934],
            [-0.67531302, -0.2168043 ],
            [ 0.97113521,  0.44832629],
            [ 0.70657936, -0.74256946]])
   """
    np.random.seed(seed)
    P = np.ones((N,2)) - 2 * np.random.random((N,2))
    return P


def plotMLN(P, D, A, g):
    """
    """
    f = plt.figure()
    ax = plt.axes([-1, -1, 10, 10])
    plt.scatter(P[:,0], P[:,1], c = D)
    L = np.vstack((A - 3*g, A + 3*g))
    plt.plot(L[:,0], L[:,1], "r--")
    plt.show()


class PLA:
    def __init__(self):
        pass
