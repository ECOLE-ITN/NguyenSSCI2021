from __future__ import print_function
from collections import OrderedDict
import numpy as np
from typing import List
#from Component.BayesOpt import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace
from BanditOpt.HyperParameter import HyperParameter, FloatParam, IntegerParam, CategoricalParam

class ForbiddenItem(object):
    def __init__(self, var_name, left, leftvalue, right, rightvalue, add1, add1value, add2, add2value, isdiffRoot):
        self.var_name= var_name
        self.left = left
        self.leftvalue= leftvalue
        self.right=right
        self.rightvalue=rightvalue
        self.ladd1 = add1
        self.ladd1value= add1value
        self.ladd2 = add2
        self.ladd2value=add2value
        self.isdiffRoot=isdiffRoot
class Forbidden(object):
    def __init__(self):
        self.forbList = OrderedDict()

    def addForbidden(self, left: HyperParameter = None, leftvalue=None, right: HyperParameter = None, rightvalue=None,add1 = None, add1value = None, add2 = None, add2value = None):
        if not isinstance(left, HyperParameter):
            raise TypeError("Hyperparameter '%s' is not an instance of "
                            "mipego.SearchSpace" %
                            str(left))
        if not isinstance(right, HyperParameter):
            raise TypeError("Hyperparameter '%s' is not an instance of "
                            "mipego.SearchSpace" %
                            str(right))
        self._addForbidden(left, leftvalue, right, rightvalue,add1, add1value, add2, add2value)

    def _addForbidden(self, left: HyperParameter = None, leftvalue=None, right: HyperParameter = None, rightvalue=None,add1 = None, add1value = None, add2 = None, add2value = None):
        isdiffRoot = False
        if not isinstance(leftvalue, (list, tuple)):
            leftvalue = [leftvalue]
        if not isinstance(rightvalue, (list, tuple)):
            rightvalue = [rightvalue]
        var_name= str(left.var_name) + '_' + str(right.var_name)+"".join([str(x) for x in rightvalue])
        if (var_name in self.forbList):
            self._addLeftvalue(left,leftvalue,right,rightvalue)
        else:
            if(add1!=None):
                add1Name=add1.var_name
                if not isinstance(add1value, (list, tuple)):
                    add1value = [add1value]
            else:
                add1Name=None
            if(add2!=None):
                add2Name=add2.var_name
                if not isinstance(add2value, (list, tuple)):
                    add2value = [add2value]
            else:
                add2Name=None
            self.forbList[var_name] = ForbiddenItem(var_name,left.var_name, leftvalue,right.var_name, rightvalue,
                                                    add1Name, add1value, add2Name, add2value, isdiffRoot)
    def _addLeftvalue(self,left: HyperParameter = None, leftvalue=None, right: HyperParameter = None, rightvalue=None):
        var_name = str(left.var_name) + '_' + str(right.var_name) + "".join(rightvalue)
        old_value=self.forbList[var_name].leftvalue
        if(old_value!= leftvalue):
            self.forbList[var_name].leftvalue=old_value+leftvalue

if __name__ == '__main__':
    np.random.seed(1)
    C = FloatParam([-5, 5]) * 3  # product of the same space
    Anh = CategoricalParam(["SVM", "KNN"], 'alg')
    kernel = IntegerParam([1, 100], 'kernel')
    I = IntegerParam([-20, 20], 'I1')
    N = CategoricalParam(['OK', 'A', 'B', 'C', 'D', 'E'], "N")

    # I3 = I * 3
    print(Anh.sampling())
    # print(kernel.sampling())
    # print(I3.var_name)

    # print(C.sampling(1, 'uniform'))

    # cartesian product of heterogeneous spaces
    con = Forbidden()
    con.addForbidden(N,"A", Anh, "SVM")
    con.addForbidden(N, "B", Anh, "SVM")
    con.addForbidden(N, "C", Anh, "SVM")
    print(con)
