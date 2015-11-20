# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:28:50 2015

@author: Manuel Pasieka , manuel.pasieka@csf.ac.at

Homework Assignment from https://work.caltech.edu/telecourse.html

Simple Perceptron Learning Model Algorithm
"""
import sys
import errno
import random


def main(argv):
    if len(argv) < 2:
        print('Please specify the number of random Training sample Points N')
        print('Usage %s N' % argv[0])
        sys.exit(errno.EINVAL)
    
    trainingData = createTestData(N)
    

def createTestData(N):
    #Sample N random values in the range x =[-1,1], y=[-1,1], and asign them two either
    #group 0, or 1 depending if the are left of x = 0 or right of it.
    coordinates = [(-1.0 + random.random()*2.0, -1.0 + random.random()*2.0) for x in range(N)]
    points = [ (p[0], p[1], 0) if p[0] < 0.0 else (p[0], p[1], 1) for p in coordinates]
    return points
    
if __name__ == '__main__':
    main(sys.argv)