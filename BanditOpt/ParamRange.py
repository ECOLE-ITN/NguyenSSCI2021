import numpy as np
from numpy.random import randint, rand
from abc import abstractmethod
'''
class bound(object):
    def __init__(self,bound):
        self.bound=bound
class ibound(bound):
    def __init__(self,lower:int, upper:int):
        super(ibound, self).__init__(bound)
        self.lower=lower
        self.upper=upper
class fbound(bound):
    def __init__(self,lower:float, upper:float):
        super(ibound, self).__init__(bound)
        self.lower=lower
        self.upper=upper
class normalbound(bound):
    def __init__(self,mu, sigma, size=()):
        NotImplemented
        '''
class paramrange(object):
    def __init__(self, bounds, default=None, hType=None):

        if isinstance(bounds,(list, tuple, np.ndarray)):
            self.bounds = [b for b in bounds]
            self.default = default
        else:
            self.bounds = [bounds]
            self.default = bounds
        if(hType=="I"):
            if isinstance(bounds,(list, tuple, np.ndarray)) and len(bounds)==2:
                self.lower = bounds[0]
                self.upper = bounds[1]
                self.type="range"
            elif isinstance(bounds,(list, tuple, np.ndarray)) and len(bounds)>2:
                self.type="list"
                pass
            else:
                self.lower = bounds[0]
                self.upper = bounds[0]
                self.default = bounds[0]
                self.type = "list"
        elif(hType=="F"):
            if isinstance(bounds,(list, tuple, np.ndarray))  and len(bounds)==2:
                self.lower = bounds[0]
                self.upper = bounds[1]
                self.type = "range"
            else:
                self.lower = bounds
                self.upper = bounds
                self.default = bounds
                self.type = "list"

class p_paramrange(paramrange):
    def __init__(self, bounds, p=None, default=None, scalerule=None, q=None, hType=None):
        '''q:quantized, use for scalerule in [quniform, qloguniform, qnormal or qlognormal]
           If the loss function is probably more correlated for nearby integer values, then you should probably use one of the "quantized" continuous distributions,
           Example:
               uniformly without q:
                    - Returns a value uniformly between low and high
                quniform:
                    - Returns a value like round(uniform(low, high) / q) * q
        '''
        super(p_paramrange,self).__init__(bounds,default, hType)
        if (p > 1):
            raise ValueError('Wrong input, p must in float in range of [0,1]')
        self.p=p
        self.scalerule=scalerule
        self.q=q
class one_paramrange(paramrange):
    def __init__(self, bounds):
        default = None
        #scalerule = None
        super(one_paramrange,self).__init__(bounds,default)

