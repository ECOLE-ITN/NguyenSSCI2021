from __future__ import print_function
from collections import OrderedDict
from typing import List
import numpy as np
#from Component.BayesOpt import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace
from BanditOpt.HyperParameter import HyperParameter, AlgorithmChoice, FloatParam, IntegerParam, CategoricalParam
class ConditionalSpace(object):
    def __init__(self, name):
        self.name = np.array(name)
        self.conditional = OrderedDict()
        self.AllConditional=OrderedDict()

    def addMutilConditional(self, child: List[HyperParameter] = None, parent: HyperParameter = None, parent_value=None, isRoot=True):
        if isinstance(parent_value,(tuple,list)):
            parent_value=[b for b in parent_value]
        else:
            parent_value=[parent_value]
        if(set(parent_value).issubset([j for i in [x.bounds for x in parent.bounds] for j in i])):
            for achild in child:
                self.addConditional(achild, parent, parent_value,isRoot)
        else:
            raise TypeError("Hyperparameter '%s' is not in range of "
                            "--" %
                            str(parent.var_name))
    def rebuild(self):
        pass


    def addConditional(self, child: HyperParameter = None, parent: HyperParameter = None, parent_value=None, isRoot=None):
        if not isinstance(child, HyperParameter):
            raise TypeError("Hyperparameter '%s' is not an instance of "
                            "mipego.SearchSpace" %
                            str(child))
        if not isinstance(parent, HyperParameter):
            raise TypeError("Hyperparameter '%s' is not an instance of "
                            "mipego.SearchSpace" %
                            str(parent))
        if isinstance(parent_value,(tuple,list)):
            parent_value=[b for b in parent_value]
        else:
            parent_value=[parent_value]
        if(set(parent_value).issubset([j for i in [x.bounds for x in parent.bounds] for j in i])==False):
            raise TypeError("Hyperparameter '%s' is not in range of "
                            "--" %
                            str(parent.var_name))
        keyname = str(child.var_name) + '_' + str(parent.var_name)
        #All conditional for impute + bandit
        if (keyname in self.AllConditional.keys()):
            self._updateAllConditional(child, parent, parent_value)
        else:
            self._addAllConditional(child, parent, parent_value)
        #list of conditional for treezation only
        if (isRoot==None):
            if (parent.var_name in [x[0] for i,x in self.conditional.items()]):
                isRoot=False
            else:
                isRoot=True
            if isinstance(parent,AlgorithmChoice):
                isRoot=True
        #if isRoot==True:
        if (keyname in self.conditional.keys()):
            self._updateConditional(child, parent, parent_value, isRoot)
        else:
            self._addConditional(child, parent, parent_value, isRoot)
    def _updateConditional(self, child: HyperParameter = None, parent: HyperParameter = None, parent_value=None, isRoot=True) -> None:
        if isRoot==True:
            if not isinstance(parent_value, (list, tuple)):
                parent_value=list(parent_value)
            keyname = str(child.var_name) + '_' + str(parent.var_name)
            old_value= self.conditional[keyname][2]
            parent_value= parent_value + old_value
            self.conditional[keyname] = [
                child.var_name, parent.var_name,
                parent_value]
        else:
            pass
    def _addConditional(self, child: HyperParameter = None, parent: HyperParameter = None, parent_value=None, isRoot=True) -> None:
        if not isinstance(parent_value, (list, tuple)):
            parent_value=list([parent_value])
        '''keyname = str(child.var_name) + '_' + str(parent.var_name)
        self.conditional[keyname] = [child.var_name, parent.var_name, parent_value]
        '''
        #D.A: remove this part 23 042021
        #if its parent has parent, add this child to all parents of its parent
        if(parent.var_name in [x[0] for _,x in self.conditional.items()]):
            for x in [x for _,x in self.conditional.items() if x[0]==parent.var_name]:
                keyname = str(child.var_name) + '_' + str(x[1])
                if (keyname in self.conditional.keys()):
                    old_value = self.conditional[keyname][2]
                    parent_value = old_value+ list( set(old_value)-set(x[2]) )
                    self.conditional[keyname] = [
                        child.var_name, x[1],parent_value]
                else:
                    self.conditional[keyname]=[child.var_name,x[1],x[2]]
        if(isRoot==True):
            keyname = str(child.var_name) + '_' + str(parent.var_name)
            self.conditional[keyname] = [child.var_name, parent.var_name, parent_value]

        #else:
            #self.conditional[str(child.var_name[0]) + '_' + str(parent.var_name[0]) + '_' + "".join(parent_value)] = [
             #   child.var_name[0], parent.var_name[0],
             #   parent_value]
            #for value in parent_value:
                #self.conditional[str(child.var_name[0]) + '_' + str(parent.var_name[0]) + '_' + str(value)] = [
                    #child.var_name[0], parent.var_name[0], value]
    def _updateAllConditional(self, child: HyperParameter = None, parent: HyperParameter = None, parent_value=None) -> None:
        if not isinstance(parent_value, (list, tuple)):
            parent_value=list(parent_value)
        keyname = str(child.var_name) + '_' + str(parent.var_name)
        old_value= self.AllConditional[keyname][2]
        parent_value= parent_value + old_value
        self.AllConditional[keyname] = [
            child.var_name, parent.var_name,
            parent_value]
    def _addAllConditional(self, child: HyperParameter = None, parent: HyperParameter = None, parent_value=None) -> None:
        if not isinstance(parent_value, (list, tuple)):
            parent_value=list([parent_value])
        keyname = str(child.var_name) + '_' + str(parent.var_name)
        self.AllConditional[keyname] = [child.var_name, parent.var_name,parent_value]

if __name__ == '__main__':
    np.random.seed(1)
    #C = FloatParam([-5, 5]) * 3  # product of the same space
    Anh = AlgorithmChoice(["SVM", "KNN"], 'alg')
    kernel = IntegerParam([1, 100], 'kernel')
    I = IntegerParam([-20, 20], 'I1')
    N = CategoricalParam(['OK', 'A', 'B', 'C', 'D', 'E'], "N")

    # I3 = I * 3
    #print(Anh.sampling())
    # print(kernel.sampling())
    # print(I3.var_name)

    # print(C.sampling(1, 'uniform'))
    alg_namestr = CategoricalParam([("SVM", 0.4), "RF", ['LR', 'DT']], "alg_namestr")
    # cartesian product of heterogeneous spaces
    con = ConditionalSpace("weare")
    con.addMutilConditional([I, kernel], alg_namestr, "RF")
    print(con)
# cs = ConfigSpace()