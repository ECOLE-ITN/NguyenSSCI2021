from __future__ import print_function
from copy import deepcopy
from collections import OrderedDict
from typing import Union, List, Dict, Optional
from numpy.random import randint
import itertools
import numpy as np
#import mipego4ml.ConditionalSpace
from BanditOpt.Forbidden import Forbidden
from BanditOpt.ConditionalSpace import ConditionalSpace
from BanditOpt.ParamExtension import rebuild
# from .mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace
from Component.BayesOpt import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace

class ConfigSpace(object):
    def __init__(self, name: Union[str, None] = None,
                 seed: Union[int, None] = None,
                 meta: Optional[Dict] = None,
                 ) -> None:
        self.name = name
        self.meta = meta
        self.random = np.random.RandomState(seed)
        self._hyperparameters = OrderedDict()  # type: OrderedDict[str, Hyperparameter]
        self._hyperparameter_idx = dict()  # type: Dict[str, int]
        self._idx_to_hyperparameter = dict()  # type: Dict[int, str]
        self._listconditional = OrderedDict()
        self._listForbidden = OrderedDict()
        self._sampler = OrderedDict()
        self._OrgLevels = OrderedDict()
        # self.var_name = np.array_str()
        self.dim = 0
        # self._sampler_updated = OrderedDict()

    def __len__(self):
        return self.dim

    def __iter__(self):
        pass

    def add_multiparameter(self, params: List[SearchSpace]) -> List[SearchSpace]:
        # listParent =OrderedDict()
        for param in params:
            if not isinstance(param, SearchSpace):
                raise TypeError("Hyperparameter '%s' is not an instance of "
                                "mipego.SearchSpace" %
                                str(param))
        for param in params:
            self._add_singleparameter(param)

        return params

    def _add_singleparameter(self, param: SearchSpace) -> None:
        setattr(param, 'iskeep', False)
        if param.var_name[0] in self._hyperparameters:
            raise ValueError("Hyperparameter '%s' is already in the "
                             "configuration space." % param.var_name[0])
        self._hyperparameters[str(param.var_name[0])] = param
        for i, hp in enumerate(self._hyperparameters):
            if not 'var_name' in locals():
                var_name = np.array([hp])
            else:
                var_name = np.r_[var_name, np.array([hp])]
            self._hyperparameter_idx[hp] = i
        self.var_name = np.array([str(var_name)])

    def _listoutsinglebraches(self, rootnode, rootname, hpi, final, lsvarname, childeffect, lsparentname):
        i = rootnode
        hp = deepcopy(self._hyperparameters)
        name = rootname
        final = final
        # print(i)
        if (i[0] in lsvarname):
            #hpa = [(hp1[0], i[1], True) for hp1 in hp if (hp1[0] == i[0])]
            temp= hp[i[0]]
            if(isinstance(i[1], tuple)):
                temp.bounds = [i[1]]
            else:
                temp.bounds = [tuple(i[1])]
            temp.iskeep = True
            #temp.rebuild()
            temp= rebuild(temp)
            hpa=[temp]
            child_hpa = [x[1] for x in childeffect if (x[0] == (i[0] + "_" + "".join(i[1])))]

            child_node = []

            if (len(child_hpa) > 0):
                if (child_hpa[0] in lsvarname):
                    child_node = [x for x in lsparentname if x[0] in child_hpa]
                else:
                    child_node = [x for x in self._listconditional.conditional.values() if x[0] in child_hpa and x[2] == i[1]]
            else:
                hpi = hpa
                if (i[0] == name):
                    final.append(hpi)
            if (len(child_hpa) < 2):
                while (len(child_node) > 0):
                    child = child_node[0]
                    i3 = child
                    hpi = self._listoutsinglebraches(i3, name, hpi, final, lsvarname, childeffect, lsparentname)
                    hpi = hpa + hpi
                    child_node.remove(child)
                    if (i[0] == name):
                        final.append(hpi)
            else:
                count = 1
                numberChild = len(child_node)
                for child in child_node:
                    i3 = child
                    hpi = self._listoutsinglebraches(i3, name, hpi, final, lsvarname, childeffect, lsparentname)
                    count = count + 1
                    if (count < 2 or count > numberChild):
                        hpi = hpa + hpi
                if (i[0] == name):
                    final.append(hpi)
        else:
            temp = hp[i[0]]
            temp.iskeep = True
            temp=rebuild(temp)
            #hpchild = [temp]
            hpi.append(temp)
        return hpi

    def listoutAllBranches(self, lsvarname, childeffect, lsparentname) -> List[SearchSpace]:
        #hp = copy.deepcopy(self._hyperparameters)
        hp=self._hyperparameters
        temp_hpi,lsBranches = [], []
        childList = [x[1] for x in childeffect]
        lsvarname=list(lsvarname)
        lsOneNode = [x for x in hp if x not in (lsvarname + childList)]
        norelationLst=[]
        for node in lsOneNode:
            item= deepcopy(hp[node])
            item.iskeep=True
            norelationLst.append([[item]])
        lsRootNode = [x for x in lsvarname if x not in childList]
        # print(hpa)
        for root in lsRootNode:
            for item in [x for x in lsparentname if x[0] == root]:
                temp_hpi=[]
                temp_hpi = self._listoutsinglebraches(item, item[0], temp_hpi, lsBranches, lsvarname, childeffect, lsparentname)
        count,final, MixList = 1,[],[]
        for root in lsRootNode:
            tempList=[]
            for aBranch in lsBranches:
                in1Lst = False
                for item in aBranch:
                    if (item.var_name[0] == root):
                        in1Lst = True
                if (in1Lst == True):
                    tempList.append(aBranch)
            MixList.append(tempList)
        MixList=MixList+norelationLst
        #MixList
        #Forbidden:
        #1: Forbidden at module level: We change the search space
        #2: Forbidden at node/child/leaves level: check in the sampling function
        listDiffRootForb=OrderedDict()
        if (self._listForbidden!=None):
            for id, item in self._listForbidden.forbList.items():
                left=item.left
                l_group, r_group=None, None
                right = item.right
                branch_id =0
                for modules in MixList:
                    for mainBranches in modules:
                        for node in mainBranches:
                            if (node.var_name[0] == left):
                                l_group=branch_id
                            if (node.var_name[0] ==right):
                                r_group=branch_id
                    branch_id+=1
                if (l_group!=r_group):
                    item.isdiffRoot=True
                    listDiffRootForb[id]=item

        lsFinalSP = []
        FinalSP = OrderedDict()
        MixList_feasible = []
        MixList_feasible_remain = []
        if(len(MixList)>1):
            final=list(itertools.product(*MixList))
            igroup = 0
            for group in final:
                group_new = list(deepcopy(group))
                for key, value in listDiffRootForb.items():
                    item_left, item_right=None, None
                    i=0
                    for module in group_new:
                        hp_left=[(idx,sp) for (idx,sp) in enumerate(module) if sp.var_name[0]==value.left and len(set(sp.bounds[0]).intersection(value.leftvalue))>0]
                        hp_right=[(idx,sp) for (idx,sp) in enumerate(module) if sp.var_name[0]==value.right and len(set(sp.bounds[0]).intersection(value.rightvalue))>0]
                        if(len(hp_left)>0):
                            module_left=i
                            index_left, item_left=hp_left[0]
                        if(len(hp_right)>0):
                            module_right=i
                            index_right,item_right=hp_right[0]
                            print(index_right)
                        i+=1
                    if (item_left!=None and item_right!=None):
                        sp_bound_left = item_left.bounds[0]
                        sp_bound_left = tuple(set(sp_bound_left) - set(value.leftvalue))
                        if (len(sp_bound_left) > 0):
                            item_left.bounds[0]=sp_bound_left
                            item_left=rebuild(item_left)
                            group_new[module_left][index_left]=item_left
                        else:
                            left_childs=[ke[0] for ke in self._listconditional.conditional.values() if ke[1]==item_left.var_name[0]]
                            group_new[module_left].pop(index_left)
                            if (len(left_childs)>0):
                                left_child_idx=[idx for (idx,x) in enumerate(group_new[module_left]) if x.var_name[0] in left_childs]
                                for child in left_child_idx:
                                    group_new[module_left].pop(child)
                        """sp_bound_right=item_right.bounds[0]
                        sp_bound_right_remain = tuple(set(sp_bound_right)-set(value.rightvalue))
                        ##ADD new module in list
                        if(len(sp_bound_right_remain)>0):
                            item_right_remain=deepcopy(item_right)
                            item_right_remain.bounds[0]=sp_bound_right_remain
                            item_right_remain=rebuild(item_right_remain)
                            group_remain = deepcopy(group)
                            group_remain[module_right][index_right]=item_right_remain
                            final.append(group_remain)
                            #MixList_feasible_remain.append(group_remain)
                        item_right.bounds[0] = value.rightvalue
                        item_right = rebuild(item_right)"""
                        group_new[module_right][index_right] = item_right
                final[igroup]=group_new
                #MixList_feasible.append(module)
                igroup+=1
            for searchSpace in final:
                for group in searchSpace:
                    for item in group:
                        if (item.iskeep == True):
                            FinalSP[item.var_name[0]] = item
                            if 'space' not in locals():
                                space = item
                            else:
                                space = space + item
                lsFinalSP.append(space)
                del space
        elif(len(MixList)==1):
            final=list(MixList)
            for searchSpace in final:
                for group in searchSpace:
                    for item in group:
                        if (item.iskeep == True):
                            FinalSP[item.var_name[0]] = item
                            if 'space' not in locals():
                                space = item
                            else:
                                space = space + item
                    lsFinalSP.append(space)
                    del space
        else:
            pass
        """
        
        for key, value in listDiffRootForb.items():
            for sp in lsFinalSP:
                try:
                    leftIndex=sp.var_name.index(value.left)
                    rightIndex = sp.var_name.index(value.right)
                except:
                    leftIndex=None
                    rightIndex=None
                if(leftIndex != None and rightIndex !=None):
                    intersectionLeft= set(sp.bounds[leftIndex]).intersection(value.leftvalue)
                    intersectionRight=set(sp.bounds[rightIndex]).intersection(value.rightvalue)
                    if (len(intersectionLeft)>0 and len(intersectionRight)>0 ):
                        sp_bound_left= sp.bounds[leftIndex]
                        sp_bound_left=tuple(set(sp_bound_left)-set(intersectionLeft))
                        if (len(sp_bound_left)>0):
                            sp.bounds[leftIndex] = sp_bound_left
                            sp.levels[leftIndex] =sp_bound_left
                            sp._n_levels[leftIndex]=len(sp_bound_left)
                        else:
                            sp.var_name.pop(leftIndex)
                            sp.bounds.pop(leftIndex)
                            sp.C_mask=np.delete(sp.C_mask,[leftIndex])
                            sp.N_mask=np.delete(sp.N_mask,[leftIndex])
                            sp.O_mask=np.delete(sp.O_mask,[leftIndex])
                            sp.dim=sp.dim-1
                            sp.id_C = [id for id in sp.id_C if id<leftIndex] + [id-1 for id in sp.id_C if id>leftIndex]
                            sp.id_N = [id for id in sp.id_N if id < leftIndex] + [id -1 for id in sp.id_N if
                                                                                  id > leftIndex]
                            sp.id_O = [id for id in sp.id_O if id < leftIndex] + [id -1 for id in sp.id_O if
                                                                                  id > leftIndex]
                            sp.levels.pop(leftIndex)
                            sp.name.pop(leftIndex)
                            sp.var_type.pop(leftIndex)
                            sp._n_levels.pop(leftIndex)
                            #sp.set_levels()

                        print(intersectionLeft,intersectionRight)
            """
        return lsFinalSP

    def combinewithconditional(self, cons: ConditionalSpace = None, forb: Forbidden = None, ifAllSolution=True) -> List[SearchSpace]:
        self._listconditional=cons
        self._listForbidden=forb
        listParam = OrderedDict()
        ordNo = 0
        lsParentName,lsChildEffect, lsFinalSP,lsVarNameinCons  = [],[],[],[]
        for i, param in self._hyperparameters.items():
            #lsVarName.append(i)
            listParam[i] = param.bounds[0]
            if len(param.id_N) >= 1:
                self._OrgLevels[ordNo] = param.bounds[0]
                ordNo += 1
        self.dim = ordNo
        for i, con in cons.conditional.items():
            if con[0] not in listParam.keys():
                raise TypeError("Hyperparameter '%s' is not exists in current ConfigSpace" %
                                str(con[0]))
            else:
                if(all(i in list(listParam[con[1]]) for i in con[2])==False):
                #if con[2] not in listParam[con[1]]:
                    raise TypeError("Value  '%s' doesnt exists" %
                                    str(con[2]))
            if con[1] not in listParam.keys():
                raise TypeError("Hyperparameter '%s' is not exists in current ConfigSpace" %
                                str(con[1]))
            if ([con[1], con[2]] not in lsParentName):
                lsParentName.append([con[1], con[2]])
            if(con[1] not in lsVarNameinCons):
                lsVarNameinCons.append(con[1])
            lsChildEffect.append([str(con[1]) + "_" + "".join(con[2]), con[0]])
        #lsParentName = [t for t in (set(tuple(i) for i in lsParentName))]
        #lsVarNameinCons = np.unique(np.array(lsVarNameinCons))
        ##List out the branhces which have values with no conditional
        if (ifAllSolution == True):
            for vName in lsVarNameinCons:
                itemValues = self._hyperparameters[vName].bounds[0]
                itemThisNode= [x[1] for x in lsParentName if x[0] ==vName]
                item_noCons=[]
                if (len(itemThisNode)>0):
                    itemThisNode2=[]
                    for item in itemThisNode:
                        itemThisNode2+=item

                    item_noCons = [x for x in itemValues if x not in itemThisNode2]
                # print(noCon)
                if(len(item_noCons)>0):
                    if(len(item_noCons)<2):
                        lsParentName.append([vName, item_noCons[0]])
                    else:
                        lsParentName.append([vName, list(item_noCons)])
                    #','.join([str(elem) for elem in noCon])
                #for a3 in noCon:
                #    lsParentName.append(tuple([a1, a3]))
        lsSearchSpace = self.listoutAllBranches(lsVarNameinCons, lsChildEffect, lsParentName)
        return lsSearchSpace
    #def _checkForbidden(self, lsSearchSpace):


if __name__ == '__main__':
    np.random.seed(1)
    cs = ConfigSpace()

    con = ConditionalSpace("test")

    dataset = NominalSpace(["anh"], "dataset")
    alg_name = NominalSpace(['SVM', 'LinearSVC', 'RF', 'DTC', 'KNN', 'Quadratic'], 'alg_name')
    # dataset = NominalSpace( [datasetStr],"dataset")
    cs.add_multiparameter([dataset, alg_name])
    ##module1
    ####Missingvalue
    missingvalue = NominalSpace(['imputer', 'fillna','fillNB'], 'missingvalue')
    strategy = NominalSpace(["mean", "median", "most_frequent", "constant"], 'strategy')
    cs.add_multiparameter([missingvalue, strategy])
    con.addConditional(strategy, missingvalue, ['imputer'])
    ####ENCODER
    encoder = NominalSpace(['OneHotEncoder', 'dummies'], 'encoder')
    OneHotEncoder_isUse = NominalSpace([True, False], 'isUse')
    dummy_na = NominalSpace([True, False], 'dummy_na')
    drop_first = NominalSpace([True, False], 'drop_first')
    cs.add_multiparameter([encoder, OneHotEncoder_isUse, dummy_na, drop_first])
    con.addConditional(OneHotEncoder_isUse, encoder, ['OneHotEncoder'])
    con.addMutilConditional([dummy_na, drop_first], encoder, ['dummies'])
    ###ReScaling
    rescaling = NominalSpace(['MinMaxScaler', 'StandardScaler', 'RobustScaler'], 'rescaling')
    ####IMBALANCED
    random_state = NominalSpace([27], 'random_state')
    aLeave = NominalSpace(["A","B","C"], 'aLeave')
    bleave= NominalSpace(["D","E","F"], 'bLeave')
    cs.add_multiparameter([rescaling, random_state,aLeave, bleave])
    con.addConditional(aLeave,strategy,["mean", "median"])
    con.addConditional(bleave,aLeave,["A","B","C"])
    # MODULE3
    # SVM
    probability = NominalSpace(['True', 'False'], 'probability')
    C = ContinuousSpace([1e-2, 100], 'C')
    kernel = NominalSpace(["linear", "rbf", "poly", "sigmoid"], 'kernel')
    coef0 = ContinuousSpace([0.0, 10.0], 'coef0')
    degree = OrdinalSpace([1, 5], 'degree')
    shrinking = NominalSpace(['True', 'False'], "shrinking")
    gamma = NominalSpace(['auto', 'value'], "gamma")
    gamma_value = ContinuousSpace([1e-2, 100], 'gamma_value')
    cs.add_multiparameter([probability, C, kernel, coef0, degree, shrinking, gamma, gamma_value])
    con.addMutilConditional([probability, C, kernel, coef0, degree, shrinking, gamma, gamma_value], alg_name, ['SVM'])
    # 'name': 'LinearSVC',
    penalty = NominalSpace(["l1", "l2"], 'penalty')
    # "loss" : hp.choice('loss',["hinge","squared_hinge"]),
    dual = NominalSpace([False], 'dual')
    tol = ContinuousSpace([0.0, 1], 'tol')
    multi_class = NominalSpace(['ovr', 'crammer_singer'], 'multi_class')
    fit_intercept = NominalSpace([False], 'fit_intercept')
    C_Lin = ContinuousSpace([1e-2, 100], 'C_Lin')
    cs.add_multiparameter([penalty, dual, tol, multi_class, fit_intercept, C_Lin])
    con.addMutilConditional([penalty, dual, tol, multi_class, fit_intercept, C_Lin], alg_name, ['LinearSVC'])
    # elif (alg_nameStr == "RF"):
    n_estimators = OrdinalSpace([5, 2000], "n_estimators")
    criterion = NominalSpace(["gini", "entropy"], "criterion")
    max_depth = OrdinalSpace([10, 200], "max_depth")
    max_features = NominalSpace(['auto', 'sqrt', 'log2', 'None'], "max_features")
    min_samples_split = OrdinalSpace([2, 200], "min_samples_split")
    min_samples_leaf = OrdinalSpace([2, 200], "min_samples_leaf")
    bootstrap = NominalSpace([True, False], "bootstrap")
    class_weight = NominalSpace(['balanced', 'None'], "class_weight")
    cs.add_multiparameter(
        [n_estimators, criterion, max_depth, max_features, min_samples_leaf, min_samples_split, bootstrap,
         class_weight])
    con.addMutilConditional([n_estimators, criterion, max_depth, max_features, min_samples_leaf, min_samples_split,
                             bootstrap, class_weight], alg_name, ['RF'])
    # 'name': 'DTC',
    splitter = NominalSpace(['best', 'random'], "splitter")
    criterion_dtc = NominalSpace(["gini", "entropy"], 'criterion_dtc')
    max_depth_dtc = OrdinalSpace([10, 200], 'max_depth_dtc')
    max_features_dtc = NominalSpace(['auto', 'sqrt', 'log2', 'None'], 'max_features_dtc')
    min_samples_split_dtc = OrdinalSpace([2, 200], 'min_samples_split_dtc')
    min_samples_leaf_dtc = OrdinalSpace([2, 200], 'min_samples_leaf_dtc')
    class_weight_dtc = NominalSpace(['balanced', 'None'], "class_weight_dtc", )
    # ccp_alpha = ContinuousSpace([0.0, 1.0],'ccp_alpha')
    cs.add_multiparameter([splitter, criterion_dtc, max_depth_dtc, max_features_dtc, min_samples_split_dtc,
                           min_samples_leaf_dtc, class_weight_dtc])
    con.addMutilConditional([splitter, criterion_dtc, max_depth_dtc, max_features_dtc, min_samples_split_dtc,
                             min_samples_leaf_dtc, class_weight_dtc], alg_name, ['DTC'])
    # elif (alg_nameStr == 'KNN'):
    n_neighbors = OrdinalSpace([5, 200], "n_neighbors")
    weights = NominalSpace(["uniform", "distance"], "weights")
    algorithm = NominalSpace(['auto', 'ball_tree', 'kd_tree', 'brute'], "algorithm")
    leaf_size = OrdinalSpace([1, 200], "leaf_size")
    p = OrdinalSpace([1, 200], "p")
    metric = NominalSpace(['euclidean', 'manhattan', 'chebyshev', 'minkowski'], "metric")
    # p_sub_type = name
    cs.add_multiparameter([n_neighbors, weights, algorithm, leaf_size, p, metric])
    con.addMutilConditional([n_neighbors, weights, algorithm, leaf_size, p, metric], alg_name, ['KNN'])

    # 'name': 'Quadratic',
    reg_param = ContinuousSpace([1e-2, 1], 'reg_param')
    store_covariance = NominalSpace(['True', 'False'], 'store_covariance')
    tol_qua = ContinuousSpace([0.0, 1], 'tol_qua')
    cs.add_multiparameter([reg_param, store_covariance, tol_qua])
    con.addMutilConditional([reg_param, store_covariance, tol_qua], alg_name, ['Quadratic'])
    #lsSpace = cs.combinewithconditional(con, ifAllSolution=True)
    forb = Forbidden()
    #con.addConditional(aLeave, strategy, ["mean", "median"])
    forb.addForbidden(strategy,"mean",alg_name,["SVM","DTC"])
    forb.addForbidden(aLeave, ["A","B"], alg_name, "SVM")
    forb.addForbidden(aLeave,["A",'B',"C"],alg_name,"RF")
    lsSpace = cs.combinewithconditional(con,forb, ifAllSolution=True)
    # lsSpace = cs.combinewithconditional(con)
    # print(lsSpace.sampling(10))
    orgDim = len(cs)
    # space1 = [Anh, kernel, I, Test]
    for ls in lsSpace:
        print("ratio:", float(len(ls) / orgDim))
        print(ls.sampling())
    # space = N * kernel * Anh * I
    # print(space.sampling(10))
    # from mipego import mipego

    # print((C * 2).var_name)
    # print((N * 3).sampling(2))