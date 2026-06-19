'''
this file will implement the NON linear Hard margin SVM,
the process/algorithm of SMO optimization will not change, 
just that instead of explicit weights we will have kernel functions.
'''
import numpy as np

class HardMarginKernelSVM:
    def __init__(self,kernel,tolerance:float,eps:float,max_passes:int):
        self.kernel=kernel
        self.tol=tolerance # used for KKT conditions floating point
        self.eps=eps       # used for identifying meangiful updates
        self.max_passes=max_passes # how many times is it allowed to iteratre over data set with out updating
        self.bias=0.0

        self.alpha=None
        self.X=None
        self.y=None
