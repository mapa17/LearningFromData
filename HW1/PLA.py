# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:53:05 2015

@author: Manuel Pasieka , manuel.pasieka@csf.ac.at

Homework Assignment from https://work.caltech.edu/telecourse.html

Simple Perceptron Linear Classifier
"""

import numpy as np
import matplotlib.pyplot as plt


def randomLine(seed=None):
    """
    Generates Point of Reference A and a direction g

    >>> randomLine(seed=1)
    (array([ 0.16595599, -0.44064899]), array([ 0.9299364 ,  0.36772039]))
    >>> A, g = randomLine(seed=1)
    >>> A
    array([ 0.16595599, -0.44064899])
    """
    if seed:
        np.random.seed(seed)
    A = np.ones((1, 2)) - 2 * np.random.random((1, 2))
    A = A[0]
    g = np.ones((1, 2)) - 2 * np.random.random((1, 2))
    g = g[0]
    g = g / np.linalg.norm(g)
    return A, g


def randomPoints(N, seed=None):
    """
    Get N 2D vectors ranging from (-1, 1)
    Vectors are a np.array of size (N, 2)

    >>> randomPoints(10, seed=13)
    array([[-0.55540482,  0.52491756],
            [-0.64855707, -0.9314984 ],
            [-0.94520223,  0.09310151],
            [-0.21808493, -0.55105303],
            [-0.28322669, -0.44403646],
            [ 0.92992695,  0.40310106],
            [ 0.88297502, -0.71412189],
            [ 0.25429194, -0.3596959 ],
            [ 0.4874401 ,  0.30483757],
            [ 0.98117446,  0.28333243]])
   """
    if seed:
        np.random.seed(seed)
    P = np.ones((N, 2)) - 2 * np.random.random((N, 2))
    return P


class PLA(object):
    """
    Perceptron Learning Algorithm

    >>> testPoints = randomPoints(100, seed=10)
    >>> testClasses = PLA.boundaryCheck(testPoints, [0, 0], [0, 1])
    >>> M = PLA()
    >>> M.train(testPoints, testClasses, seed=23)
    (0.98999999999999999, array([-0.06879456, -0.01460839]), array([-0.0094662 ,  0.99995519]))
    >>> P = randomPoints(1000)
    >>> T = PLA.boundaryCheck(P, [0, 0], [0, 1])
    >>> M.test(P, T)
    0.96299999999999997
    >>> #M.plot(P, T)
    """
    def __init__(self, A=[0, 0], g=[1, 0]):
        self.A = A
        self.g = g

    def train(self, P, T, N=100, correct=0.95, seed=None, debug=False):
        pC, self.A, self.g = self.__learn(P, T, N=N, correct=correct, A=self.A, g=self.g, seed=seed, debug=debug)
        return(pC, self.A, self.g)

    def test(self, P, T):
        correct, _, _, _ = self.__testModel(P, T, self.A, self.g)
        return correct

    def classify(self, P):
        return(PLA.__boundaryCheck(P, self.A, self.g))

    @staticmethod
    def __learn(P, T, N, correct=1.0, A=[0, 0], g=[0, 0], seed=None, debug=False):
        n = 0
        pC = 0
        best, _, _, _ = PLA.__testModel(P, T, A, g)
        bA = A
        bg = g
        if seed:
            np.random.seed(seed)

        while((n < N) and (pC < correct)):
            A, g = randomLine()
            pC, G1, G2, C = PLA.__testModel(P, T, A, g)
            if debug:
                print('Iteration %d, %f%%' % (n, pC))
            if pC > best:
                if debug:
                    print('New best!')
                best = pC
                bA = A
                bg = g
            n = n + 1

        pC, _, _, _ = PLA.__testModel(P, T, bA, bg)
        if debug:
            print('Finished after %d Iterations\nBest solution has %f%% correct classified points!\nA (%f, %f), g(%f, %f)' % (n, pC, bA[0], bA[1], bg[0], bg[1]))
            plotMLN(P, T, bA, bg)

        return(pC, bA, bg)

    @staticmethod
    def __testModel(P, T, A, g):
        """
        Test the classification for points P of Type T, with boundary (A, g)

        Returns
        % correct classification
        % Type -1 correct classification
        % Type 1 correct classification
        np.array of type boolean indication for each point its correctness
        """
        C = PLA.boundaryCheck(P, A, g)
        correct = C == T
        pC = correct.sum() / len(T)
        T1 = T < 0
        T2 = T > 0
        G1correct = correct[T1].sum() / T1.sum()
        G2correct = correct[T2].sum() / T2.sum()
        return pC, G1correct, G2correct, correct

    @staticmethod
    def boundaryCheck(P, A, g):
        """
        Checks if Points in P are on the left or right side of the line defined by A, and g

        Return a np.array, containing -1 for points (right), and 1 for points left of the line

        >>> P = randomPoints(10, seed=13)
        >>> A, g = randomLine(seed=23)
        >>> D = PLA.boundaryCheck(P, A, g)
        >>> D
        array([-1.,  1., -1., -1., -1., -1., -1., -1., -1., -1.])
        """
        return np.sign(np.cross(A - P, g))

    def plot(self, P, T):
        """
        Plot Points and Line, color Points depending on D
        """
        A = self.A
        g = self.g

        f = plt.figure()
        ax = plt.axes()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        co = ['r' if x < 0 else 'b' for x in T]
        plt.scatter(P[:, 0], P[:, 1], c=co, label='Points', edgecolors=co)

        L = np.vstack((A - 3 * g, A + 3 * g))
        plt.plot(L[:, 0], L[:, 1], "g--", label='Boundary')
        ax.arrow(A[0], A[1], g[0] / 5.0, g[1] / 5.0, color='c', label=format('A(%f, %f) -> (%f, %f)' % (A[1], A[1], g[0], g[1])))
        ax.legend()
        pC, _, _, _ = self.__testModel(P, T, A, g)
        plt.suptitle(format('%.2f%% Coverage, Boundary (%f, %f) -> (%f, %f)' % (pC, A[1], A[1], g[0], g[1])))
        plt.show()
        return f
