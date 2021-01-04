from __future__ import print_function
from collections import OrderedDict
from typing import List
import numpy as np
from Component.BayesOpt import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace

class ConditionalSpace(object):
    def __init__(self, name):
        self.name = np.array(name)
        self.conditional = OrderedDict()
        self.AllConditional=OrderedDict()

    def addMutilConditional(self, child: List[SearchSpace] = None, parent: SearchSpace = None, parent_value=None, isRoot=True):
        for achild in child:
            self.addConditional(achild, parent, parent_value,isRoot)
    def rebuild(self):
        pass


    def addConditional(self, child: SearchSpace = None, parent: SearchSpace = None, parent_value=None, isRoot=None):
        if not isinstance(child, SearchSpace):
            raise TypeError("Hyperparameter '%s' is not an instance of "
                            "mipego.SearchSpace" %
                            str(child))
        if not isinstance(parent, SearchSpace):
            raise TypeError("Hyperparameter '%s' is not an instance of "
                            "mipego.SearchSpace" %
                            str(parent))
        keyname = str(child.var_name[0]) + '_' + str(parent.var_name[0])
        #All conditional for impute + bandit
        if (keyname in self.AllConditional.keys()):
            self._updateAllConditional(child, parent, parent_value)
        else:
            self._addAllConditional(child, parent, parent_value)
        #list of conditional for treezation only
        if (isRoot==None):
            if (child.var_name[0] in [x[0] for i,x in self.conditional.items()]):
                isRoot=False
            else:
                isRoot=True
        #if isRoot==True:
        if (keyname in self.conditional.keys()):
            self._updateConditional(child, parent, parent_value, isRoot)
        else:
            self._addConditional(child, parent, parent_value, isRoot)
    def _updateConditional(self, child: SearchSpace = None, parent: SearchSpace = None, parent_value=None, isRoot=True) -> None:
        if isRoot==True:
            if not isinstance(parent_value, (list, tuple)):
                parent_value=list(parent_value)
            keyname = str(child.var_name[0]) + '_' + str(parent.var_name[0])
            old_value= self.conditional[keyname][2]
            parent_value= parent_value + old_value
            self.conditional[keyname] = [
                child.var_name[0], parent.var_name[0],
                parent_value]
        else:
            pass
    def _addConditional(self, child: SearchSpace = None, parent: SearchSpace = None, parent_value=None, isRoot=True) -> None:
        if not isinstance(parent_value, (list, tuple)):
            parent_value=list([parent_value])
        keyname = str(child.var_name[0]) + '_' + str(parent.var_name[0])

        #if its parent has parent, add this child to all parents of its parent
        if(parent.var_name[0] in [x[0] for _,x in self.conditional.items()]):
            for x in [x for _,x in self.conditional.items() if x[0]==parent.var_name[0]]:
                keyname = str(child.var_name[0]) + '_' + str(x[1])
                if (keyname in self.conditional.keys()):
                    old_value = self.conditional[keyname][2]
                    parent_value = old_value+ list( set(old_value)-set(x[2]) )
                    self.conditional[keyname] = [
                        child.var_name[0], x[1],parent_value]
                else:
                    self.conditional[keyname]=[child.var_name[0],x[1],x[2]]
        if(isRoot==True):
            self.conditional[keyname] = [child.var_name[0], parent.var_name[0], parent_value]
        #else:
            #self.conditional[str(child.var_name[0]) + '_' + str(parent.var_name[0]) + '_' + "".join(parent_value)] = [
             #   child.var_name[0], parent.var_name[0],
             #   parent_value]
            #for value in parent_value:
                #self.conditional[str(child.var_name[0]) + '_' + str(parent.var_name[0]) + '_' + str(value)] = [
                    #child.var_name[0], parent.var_name[0], value]
    def _updateAllConditional(self, child: SearchSpace = None, parent: SearchSpace = None, parent_value=None) -> None:
        if not isinstance(parent_value, (list, tuple)):
            parent_value=list(parent_value)
        keyname = str(child.var_name[0]) + '_' + str(parent.var_name[0])
        old_value= self.AllConditional[keyname][2]
        parent_value= parent_value + old_value
        self.AllConditional[keyname] = [
            child.var_name[0], parent.var_name[0],
            parent_value]
    def _addAllConditional(self, child: SearchSpace = None, parent: SearchSpace = None, parent_value=None) -> None:
        if not isinstance(parent_value, (list, tuple)):
            parent_value=list([parent_value])
        keyname = str(child.var_name[0]) + '_' + str(parent.var_name[0])
        self.AllConditional[keyname] = [child.var_name[0], parent.var_name[0],parent_value]

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