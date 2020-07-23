from __future__ import print_function
from collections import OrderedDict
import numpy as np
from typing import List
from Component.BayesOpt import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace

class ForbiddenItem(object):
    def __init__(self, var_name, left, leftvalue, right, rightvalue, isdiffRoot):
        self.var_name= var_name
        self.left = left
        self.leftvalue= leftvalue
        self.right=right
        self.rightvalue=rightvalue
        self.isdiffRoot=isdiffRoot
class Forbidden(object):
    def __init__(self):
        self.forbList = OrderedDict()

    def addForbidden(self, left: SearchSpace = None, leftvalue=None, right: SearchSpace = None, rightvalue=None):
        if not isinstance(left, SearchSpace):
            raise TypeError("Hyperparameter '%s' is not an instance of "
                            "mipego.SearchSpace" %
                            str(left))
        if not isinstance(right, SearchSpace):
            raise TypeError("Hyperparameter '%s' is not an instance of "
                            "mipego.SearchSpace" %
                            str(right))
        self._addForbidden(left, leftvalue, right, rightvalue)

    def _addForbidden(self, left: SearchSpace = None, leftvalue=None, right: SearchSpace = None, rightvalue=None):
        isdiffRoot = False
        if not isinstance(leftvalue, (list, tuple)):
            leftvalue = [leftvalue]
        if not isinstance(rightvalue, (list, tuple)):
            rightvalue = [rightvalue]
        var_name= str(left.var_name[0]) + '_' + str(right.var_name[0])+"".join(rightvalue)
        if (var_name in self.forbList):
            self._addLeftvalue(left,leftvalue,right,rightvalue)
        else:
            self.forbList[var_name] = ForbiddenItem(var_name,left.var_name[0], leftvalue,right.var_name[0], rightvalue, isdiffRoot)
    def _addLeftvalue(self,left: SearchSpace = None, leftvalue=None, right: SearchSpace = None, rightvalue=None):
        var_name = str(left.var_name[0]) + '_' + str(right.var_name[0]) + "".join(rightvalue)
        old_value=self.forbList[var_name].leftvalue
        if(old_value!= leftvalue):
            self.forbList[var_name].leftvalue=old_value+leftvalue

if __name__ == '__main__':
    np.random.seed(1)
    C = ContinuousSpace([-5, 5]) * 3  # product of the same space
    Anh = NominalSpace(["SVM", "KNN"], 'alg')
    kernel = OrdinalSpace([1, 100], 'kernel')
    I = OrdinalSpace([-20, 20], 'I1')
    N = NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E'], "N")

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
