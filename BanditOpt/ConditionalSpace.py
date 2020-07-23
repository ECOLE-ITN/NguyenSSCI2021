from __future__ import print_function
from collections import OrderedDict
from typing import List
import numpy as np
from Component.BayesOpt import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace


class ConditionalSpace(object):
    def __init__(self, name):
        self.name = np.array(name)
        self.conditional = OrderedDict()

    def addMutilConditional(self, child: List[SearchSpace] = None, parent: SearchSpace = None, parent_value=None):
        for achild in child:
            self.addConditional(achild, parent, parent_value)

    def addConditional(self, child: SearchSpace = None, parent: SearchSpace = None, parent_value=None):
        if not isinstance(child, SearchSpace):
            raise TypeError("Hyperparameter '%s' is not an instance of "
                            "mipego.SearchSpace" %
                            str(child))
        if not isinstance(parent, SearchSpace):
            raise TypeError("Hyperparameter '%s' is not an instance of "
                            "mipego.SearchSpace" %
                            str(parent))
        keyname=str(child.var_name[0]) + '_' + str(parent.var_name[0])
        if (keyname in self.conditional.keys()):
            self._updateConditional(child, parent, parent_value)
        else:
            self._addConditional(child, parent, parent_value)
    def _updateConditional(self, child: SearchSpace = None, parent: SearchSpace = None, parent_value=None) -> None:
        if not isinstance(parent_value, (list, tuple)):
            parent_value=list(parent_value)
        keyname = str(child.var_name[0]) + '_' + str(parent.var_name[0])
        old_value= self.conditional[keyname][2]
        parent_value= parent_value + old_value
        self.conditional[keyname] = [
            child.var_name[0], parent.var_name[0],
            parent_value]
    def _addConditional(self, child: SearchSpace = None, parent: SearchSpace = None, parent_value=None) -> None:
        if not isinstance(parent_value, (list, tuple)):
            parent_value=list(parent_value)
        keyname = str(child.var_name[0]) + '_' + str(parent.var_name[0])
        self.conditional[keyname] = [child.var_name[0], parent.var_name[0],parent_value]
        #else:
            #self.conditional[str(child.var_name[0]) + '_' + str(parent.var_name[0]) + '_' + "".join(parent_value)] = [
             #   child.var_name[0], parent.var_name[0],
             #   parent_value]
            #for value in parent_value:
                #self.conditional[str(child.var_name[0]) + '_' + str(parent.var_name[0]) + '_' + str(value)] = [
                    #child.var_name[0], parent.var_name[0], value]


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
    con = ConditionalSpace("weare")
    con.addMutilConditional([I, kernel], Anh, "SVM")
    print(con)
# cs = ConfigSpace()