from __future__ import print_function
from copy import deepcopy
from collections import OrderedDict
from typing import Union, List, Dict, Optional
from numpy.random import randint
import itertools, collections, math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#import mipego4ml.ConditionalSpace
from BanditOpt.Forbidden import Forbidden
from BanditOpt.ConditionalSpace import ConditionalSpace
#from Component.BayesOpt.ParamExtension import rebuild
from sklearn.cluster import AgglomerativeClustering
# from .mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace
#from Component.BayesOpt import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace
from BanditOpt.HyperParameter import HyperParameter,AlgorithmChoice, FloatParam, IntegerParam, CategoricalParam
from BanditOpt.ParamRange import paramrange, p_paramrange, one_paramrange
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
    def rebuild(self):
        pass
    def add_multiparameter(self, params: List[HyperParameter]) -> List[HyperParameter]:
        # listParent =OrderedDict()
        for param in params:
            if not isinstance(param, HyperParameter):
                raise TypeError("Hyperparameter '%s' is not an instance of "
                                "BO4ML.HyperParameter" %
                                str(param.var_name))
        for param in params:
            self._add_singleparameter(param)

        return params

    def _add_singleparameter(self, param: HyperParameter) -> None:
        setattr(param, 'iskeep', False)
        if param.var_name in self._hyperparameters:
            raise ValueError("Hyperparameter '%s' is already in the "
                             "configuration space." % param.var_name)
        self._hyperparameters[str(param.var_name)] = param
        for i, hp in enumerate(self._hyperparameters):
            if not 'var_name' in locals():
                var_name = np.array([hp])
            else:
                var_name = np.r_[var_name, np.array([hp])]
            self._hyperparameter_idx[hp] = i
        self.var_name = np.array([str(var_name)])
    def _getnodechilds(self,node,childeffect,lsvarname,lsparentname):
        if (isinstance(self._hyperparameters[node[0]], (AlgorithmChoice, CategoricalParam))==False):
            return [],[]
        child_hpa = [x[1] for x in childeffect if (x[0] == (node[0] + "_" + "".join(str(e) for e in node[1])))]
        n_child=len(child_hpa)
        child_node=[]
        if (n_child > 0):
            icount = 0
            #child_node = []
            for child_hpa_i in child_hpa:
                icount += 1
                if (child_hpa_i in lsvarname):
                    childlst=[x for x in lsparentname if x[0] == child_hpa_i]
                    child_node.extend(childlst)
                else:
                    childlst=[child_hpa_i,list(self._hyperparameters[child_hpa_i].allbounds)]
                    child_node.append(childlst)
                    #childlst=[x for x in self._listconditional.conditional.values() if
                                        #x[0] == child_hpa_i and x[2] == node[1]]

        return child_hpa,child_node
    def _listoutallnode(self, node, rootname, lsvarname, childeffect,
                               lsparentname, mixlist,feeded):
        temp = deepcopy(self._hyperparameters[node[0]])
        if(temp.var_name not in feeded):
            feeded.append(temp.var_name)
        if isinstance(temp,(AlgorithmChoice, CategoricalParam)):
            frange = []
            for bound in temp.bounds:
                temp_range = HyperParameter
                _thisnode = set(node[1])
                #_thisnode=set([j for i in [x for x in node[1]] for j in i])
                if(len(_thisnode.intersection(bound.bounds))>0):
                    temp_range=bound
                    _iterbound=list(_thisnode.intersection(bound.bounds))
                    if isinstance(bound,p_paramrange):
                        temp_range.p= round(len(_iterbound) * (bound.p/len(bound.bounds)),2)
                    if (isinstance(_iterbound, (tuple,list))):
                        temp_range.bounds= [b for b in _iterbound]
                    else:
                        temp_range.bounds = [_iterbound]
                    frange.append(temp_range)
            temp.bounds=frange
            temp.allbounds = [j for i in [x.bounds for x in frange] for j in i]

        temp.iskeep = False

        this_node = [temp]
        if (isinstance(temp, (AlgorithmChoice, CategoricalParam))):
            child_hpa, child_nodes = self._getnodechilds(node, childeffect, lsvarname, lsparentname)
        else:
            child_hpa, child_nodes =[],[]
        if(len(child_nodes)>0):
            for child in child_nodes:
                child_node=self._listoutallnode(child,rootname,lsvarname, childeffect,
                               lsparentname, mixlist,feeded)
                this_node.extend(child_node)
        if(node[0]==rootname):
            mixlist.extend(this_node)
        return this_node
    def _listoutBranches4(self,node, rootname, lsvarname, childeffect,
                               lsparentname):
        mixlist,feeded,ActiveLst=[],[],[]
        _ = self._listoutallnode(node, rootname, lsvarname, childeffect,
                               lsparentname, mixlist,feeded )
        final_lst=[]
        #set_unique=set([x.var_name for x in mixlist])
        for x in feeded:
            temp = []
            for item in [s for s in mixlist if s.var_name == x]:
                temp.append(item)
            final_lst.append(temp)
        final=list(itertools.product(*final_lst))
        root = node[0]
        rootvalue=node[1]
        ActiveLst.append(node)
        lsParentName, childList=[],[]
        for i, con in self._listconditional.conditional.items():
            if ([con[1], con[2], con[0]] not in lsParentName):
                lsParentName.append([con[1], con[2], con[0]])
            if (con[0] not in childList):
                childList.append(con[0])
        finalA=dict()
        for sp in final:
            sp=deepcopy(sp)
            childs = [(x[2], x[1]) for x in lsParentName if x[0] == root and len(set(rootvalue).intersection(x[1])) > 0]
            """for item in sp:
                if(item.var_name[0]==root):
                    item.iskeep=True
                    break"""
            rootnode = [x for x in sp if x.iskeep == False and x.var_name== root][0]
            rootnode.iskeep=True
            while (len(childs)>0):
                childofChild=[]
                for child in childs:
                    item= [x for x in sp if x.iskeep==False and x.var_name==child[0]][0]
                    childvalue=list(item.allbounds)
                    #if (item.var_name[0]==child[0] and len(set(child[1]).intersection(set(item.bounds[0]))) > 0):
                    item.iskeep=True
                    #childvalue = item.bounds
                    childofChild.extend([(x[2], x[1]) for x in lsParentName if x[0] == child[0] and len(set(childvalue).intersection(set(x[1])))>0])
                    #del childs[idx]
                    #break
                childs.clear()
                if(len(childofChild)>0):
                    childs=childofChild
            temp = [x for x in sp if x.iskeep==True]
            ##remove duplicate
            temp_id="id_"
            for i in temp:
                if (isinstance(i,(AlgorithmChoice, CategoricalParam))):
                    temp_id=temp_id+i.var_name+"_"+"_".join(str(e) for e in list(i.allbounds))
                else:
                    temp_id = temp_id + i.var_name
            finalA[temp_id]=temp
        finalA=[x for x in finalA.values()]

        return finalA
    def _listoutsinglebraches3(self, node, rootname, hpi, final, lsvarname, childeffect,
                               lsparentname, pathLen, mixlist):
        temp = deepcopy(self._hyperparameters[node[0]])
        temp.iskeep = True
        child_hpa, child_nodes =self._getnodechilds(node,childeffect,lsvarname,lsparentname)
        if (len(child_nodes)>0):
            for hpa in child_hpa:
                hpa_childs= [x[1] for x in child_nodes if x[0]==hpa]
                for childs in hpa_childs:
                    for child in childs:
                        node1=[hpa,child]
                        node_value= self._listoutsinglebraches3(node1, rootname, hpi, final, lsvarname, childeffect,
                               lsparentname, pathLen, mixlist)
                        mixlist.append(node_value)

        return temp

    def _listoutsinglebraches2(self, node,rootname, hpi, final, lsvarname, childeffect,
                              lsparentname, pathLen,mixlist):
        final = final
        temp = deepcopy(self._hyperparameters[node[0]])
        temp.iskeep = True
        #hpi["_".join(node[0])] = temp
        mixlist.append(node)
        if(node in hpi.keys()):
            hpi[node]=temp
        else:
            hpi.append(temp)
        pathLen+=1
        if (node[0] in lsvarname):
            ##Add to temporary list
            if (isinstance(node[1], tuple)):
                temp.bounds = [node[1]]
            else:
                temp.bounds = [tuple(node[1])]

            ##Check if the current node has children
            child_hpa = [x[1] for x in childeffect if (x[0] == (node[0] + "_" + "".join(node[1])))]
            child_node = []
            n_child= len(child_hpa)
            #IF the current node has childs, we list out its childs
            if (n_child > 0):
                icount=0
                for child_hpa_i in child_hpa:
                    child_node = []
                    icount+=1
                    if (child_hpa_i in lsvarname):
                        child_node.extend([[x for x in lsparentname if x[0] == child_hpa_i]])
                    else:
                        child_node.extend([[x for x in self._listconditional.conditional.values() if x[0] == child_hpa_i and x[2] == node[1]]])
                    for child_id, child in enumerate(child_node):
                        hpb= self._listoutsinglebraches2(child, rootname,hpi, final, lsvarname, childeffect, lsparentname,
                                                          pathLen)
            if(node[0]==rootname):
                icount=0
                tempList = []

                for key,value in hpi.items():
                    if(icount<=pathLen):
                        thisNode = deepcopy(value)
                        tempList.append(thisNode)
                    icount+=1
                final.append(deepcopy(tempList))
            #else:
                #final.append(deepcopy(hpi[0:pathLen]))
        else:
            pass
            #hpi.append(temp)
            #pass
            #final.append(deepcopy(hpi[0:pathLen]))
        #return hpi

    def _listoutsinglebraches(self, node, rootname, hpi, final, lsvarname, childeffect,
                              lsparentname,nodeAfterRoot):
        hp = deepcopy(self._hyperparameters)
        name = rootname
        final = final
        #afterRoot = [x[1] for x in childeffect if (x[0] == (rootname + "_" + "".join(node[1])))]
        # print(i)
        if (node[0] in lsvarname):
            temp= hp[node[0]]
            if(isinstance(node[1], tuple)):
                temp.bounds = [node[1]]
            else:
                temp.bounds = [tuple(node[1])]
            temp.iskeep = True
            #temp= rebuild(temp)
            ##Add to temporary list
            hpa=[temp]
            ##Check if the current node has children
            child_hpa = [x[1] for x in childeffect if (x[0] == (node[0] + "_" + "".join(node[1])))]
            child_node = []
            n_child= len(child_hpa)
            #IF the current node has childs, we list out its childs
            if (n_child > 0):
                for child_hpa_i in child_hpa:
                    if (child_hpa_i in lsvarname):
                        child_node.extend([x for x in lsparentname if x[0] == child_hpa_i])
                    else:
                        child_node.extend([x for x in self._listconditional.conditional.values() if x[0] == child_hpa_i and x[2] == node[1]])
            elif(n_child==0):
                #if the current node has no child, we add it into the major list
                hpi.extend(hpa)
                if (node[0] == name):
                    final.append(hpi)
                    return hpi
            #If the case: the current node has only 1 child
            if (n_child ==1):
                while (len(child_node)>0):
                    child = child_node[0]
                    if (child in (nodeAfterRoot)):
                        hpi.clear()
                    hpi = self._listoutsinglebraches(child, name, hpi, final, lsvarname, childeffect, lsparentname, nodeAfterRoot)
                    hpi.extend(hpa)
                    child_node.remove(child)
                    if (child in (nodeAfterRoot)):
                        final.append(hpi)
                        return hpi
            elif(n_child>1):
                count = 1
                temp_hpi=[]
                for child in child_node:
                    hpi = self._listoutsinglebraches(child, name, hpi, final, lsvarname, childeffect, lsparentname, nodeAfterRoot)
                    temp_hpi.append([child, name, hpi])
                    count = count + 1
                    ##TO DO: BUG
                    if (count > n_child):
                        hpi = hpa + hpi
                if (node[0] == name):
                    final.append(hpi)
        else:
            temp = hp[node[0]]
            temp.iskeep = True

            #hpchild = [temp]
            hpi.append(temp)
        return hpi
    def _clustering(self,final,sp_cluster):

        header=[x.var_name for x in self._hyperparameters.values() if isinstance(x,(AlgorithmChoice, CategoricalParam))]
        notcount=[x.var_name for x in self._hyperparameters.values() if isinstance(x,(AlgorithmChoice, CategoricalParam))==False]
        le = LabelEncoder()
        LstEnc = dict()
        for i in header:
            item = self._hyperparameters[i]
            le.fit(item.allbounds)
            LstEnc[i] = le.classes_
        df=[]
        for idx,sp in enumerate(final):
            itemarr= [idx]
            for item in header:
                try:
                    #i = sp[item]
                    i=[x for sublist in sp for x in sublist if x.var_name == item][0]
                    ivalue=i.allbounds
                    le.classes_ = LstEnc[item]
                    ivalue=tuple(le.transform(ivalue))
                except:
                    ivalue=None
                itemarr.append(ivalue)
            df.append(itemarr)
        header1 = ["id"]
        header1.extend(header)

        df=pd.DataFrame(df,columns=header1)
        dfReturn=pd.DataFrame(df['id'])
        for col in header:

            #dff = pd.DataFrame()
            dff=[]
            for idx,x in enumerate(df[col]):
                alst = dict()
                alst['idx']=idx
                if (isinstance(x,tuple) or isinstance(x,list)):
                    for i in x:
                        alst[col+"_"+str(i)]=1
                dff.append(alst)
                #dff=dff.append(alst, ignore_index=True)
            #dff = pd.DataFrame(dff)
            #dfReturn = pd.concat([dfReturn[:], dff[:]], axis=1)
            dfReturn=dfReturn.join(pd.DataFrame(dff).set_index('idx'),on='id')
        dfReturn = dfReturn.fillna(0)
        ncluster=sp_cluster
        complete = AgglomerativeClustering(n_clusters=ncluster, linkage='complete')
        # Fit & predict
        # Make AgglomerativeClustering fit the dataset and predict the cluster labels
        complete_pred = complete.fit_predict(dfReturn)
        lsReturn=[]
        for idx, i in enumerate(complete_pred):
            lsReturn.append([i,final[idx]])
        newFinal=[]
        for i in range(ncluster):
            temp=dict()
            for sp in [x[1] for x in lsReturn if x[0]==i]:
                for x in [x for sublist in sp for x in sublist]:
                    if (x.var_name in temp.keys()):
                        lastvalue=temp[x.var_name].allbounds
                        thisvalue=x.allbounds
                        diff= tuple(set(thisvalue)-set(lastvalue))
                        if(len(diff)>0):
                            temp[x.var_name].allbounds=lastvalue+diff
                    else:
                        temp[x.var_name]=x
            '''for _,x in temp.items():
                x=rebuild(x)'''
            newFinal.append(temp)
        return newFinal
    def listoutAllBranches(self, lsvarname, childeffect, lsparentname,sp_cluster=0) -> List[HyperParameter]:
        #hp = copy.deepcopy(self._hyperparameters)
        hp=self._hyperparameters
        temp_hpi,lsBranches = [], []
        childList = [x[1] for x in childeffect]
        lsvarname=list(lsvarname)
        lsOneNode = [x for x in hp if x not in (lsvarname + childList)]
        norelationLst=[]
        for node in lsOneNode:
            _thisList=[]
            for _value in [x[1] for x in lsparentname if x[0]==node]:
                item = deepcopy(hp[node])
                _newbounds = []
                for _x in [x for x in item.bounds if len(set(_value).intersection(x.bounds))>0]:
                    _temp=deepcopy(_x)
                    _temp.bounds=list(set(_value).intersection(_temp.bounds))
                    _newbounds.append(_temp)
                item.iskeep=True
                item.bounds=_newbounds
                item.allbounds=[j for i in [x.bounds for x in _newbounds] for j in i]
                _thisList.append([item])
            norelationLst.append(_thisList)
        lsRootNode = [x for x in lsvarname if x not in childList]
        # print(hpa)
        for root in lsRootNode:
            for item in [x for x in lsparentname if x[0] == root]:
                finalA=self._listoutBranches4(item,root,lsvarname, childeffect, lsparentname)
                lsBranches.extend(finalA)
        count,final, MixList = 1,[],[]
        for root in lsRootNode:
            tempList=[]
            for aBranch in lsBranches:
                in1Lst = False
                for item in aBranch:
                    if (item.var_name == root):
                        in1Lst = True
                if (in1Lst == True):
                    tempList.append(aBranch)
            MixList.append(tempList)
        MixList=MixList+norelationLst
        #lsRootNode.extend(lsOneNode)
        #Forbidden:
        #1: Forbidden at module levels: We change the search space
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
                            if (node.var_name == left):
                                l_group=branch_id
                            if (node.var_name ==right):
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
            tobedel=[]
            igroup = 0
            for group in final:
                group_new = list(deepcopy(group))
                isDelete = False
                for key, value in listDiffRootForb.items():
                    item_left, item_right=None, None
                    isBothRoot=False
                    if((value.left in lsRootNode) and (value.right in lsRootNode)):
                        isBothRoot=True
                    else:
                        continue
                    i=0
                    for module in group_new:
                        #hp_left=[(idx,sp) for (idx,sp) in enumerate(module) if sp.var_name==value.left and len(set(sp.allbounds).intersection(value.leftvalue))>0]
                        #hp_right=[(idx,sp) for (idx,sp) in enumerate(module) if sp.var_name==value.right and len(set(sp.allbounds).intersection(value.rightvalue))>0]
                        hp_left = [(idx, sp) for (idx, sp) in enumerate(module) if sp.var_name == value.left and len(
                            set(sp.allbounds)-set(value.leftvalue)) == 0]
                        hp_right = [(idx, sp) for (idx, sp) in enumerate(module) if sp.var_name == value.right and len(
                            set(sp.allbounds)-set(value.rightvalue)) == 0]
                        if(len(hp_left)>0):
                            module_left=i
                            index_left, item_left=hp_left[0]
                        if(len(hp_right)>0):
                            module_right=i
                            index_right,item_right=hp_right[0]
                            #print(index_right)
                        i+=1
                    if (item_left!=None and item_right!=None):
                        sp_bound_left = item_left.allbounds
                        sp_bound_left_remain = tuple(set(sp_bound_left) - set(value.leftvalue))
                        sp_bound_right = item_right.allbounds
                        sp_bound_right = tuple(set(sp_bound_right) - set(value.rightvalue))
                        if (len(sp_bound_right) < 1 and len(sp_bound_left_remain) < 1 and isBothRoot == True):
                            isDelete = True
                        if (len(sp_bound_right) > 0):
                            #item_left.allbounds=sp_bound_left
                            frange=[]
                            for bound in item_right.bounds:
                                if (len(set(sp_bound_right).intersection(bound.bounds)) > 0):
                                    temp_range = bound
                                    if isinstance(bound, p_paramrange):
                                        temp_range.p = round(len(sp_bound_right) * (bound.p / len(bound.bounds)), 2)
                                    if (isinstance(sp_bound_right, (tuple, list))):
                                        temp_range.bounds = [b for b in sp_bound_right]
                                    else:
                                        temp_range.bounds = [sp_bound_right]
                                    frange.append(temp_range)
                            item_right.bounds=frange
                            item_right.allbounds = [j for i in [x.bounds for x in frange] for j in i]
                            item_right.default= item_right.default if item_right.default in item_right.allbounds else item_right.allbounds[0]
                            #item_left=rebuild(item_left)
                            group_new[module_right][index_right]=item_right
                            _del_right_childs=[ke[0] for ke in self._listconditional.conditional.values() if ke[1]==item_right.var_name
                                                 and len(set(value.rightvalue)-set(ke[2]))==0]
                            if len(_del_right_childs)>0:
                                lChild_del = []
                                while (len(_del_right_childs) > 0):
                                    for right_child in _del_right_childs:
                                        _del_right_childs.extend([ke[0] for ke in self._listconditional.conditional.values() if
                                                             ke[1] == right_child])
                                        lChild_del.append(right_child)
                                        _del_right_childs.remove(right_child)
                                lIndex_del=[idx for (idx, x) in enumerate(group_new[module_right]) if
                                                   x.var_name in lChild_del]
                                for index in sorted(lIndex_del, reverse=True):
                                    del group_new[module_right][index]
                        else:
                            if (isDelete==False):
                                right_childs=[ke[0] for ke in self._listconditional.conditional.values() if ke[1]==item_right.var_name
                                             and len(set(item_right.allbounds)-set(ke[2]))==0]
                                lIndex_del=[index_right]
                                lChild_del=[]
                                #group_new[module_left].pop(index_left)
                                while(len(right_childs)>0):
                                    for right_child in right_childs:
                                        right_childs.extend([ke[0] for ke in self._listconditional.conditional.values() if
                                         ke[1] == right_child])
                                        lChild_del.append(right_child)
                                        right_childs.remove(right_child)
                                lIndex_del.extend([idx for (idx, x) in enumerate(group_new[module_right]) if
                                                  x.var_name in lChild_del])
                                for index in sorted(lIndex_del, reverse=True):
                                    del group_new[module_right][index]

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
                        #group_new[module_right][index_right] = item_right
                if (isDelete==False):
                    final[igroup]=group_new
                else:
                    tobedel.append(igroup)
                #MixList_feasible.append(module)
                igroup+=1
            for index in sorted(tobedel, reverse=True):
                del final[index]
            if (sp_cluster>0 and len(final)>sp_cluster):
                final=self._clustering(final,sp_cluster)
                for group in final:
                    #defaults = []
                    space = []
                    for _,item in group.items():
                        #if (item.iskeep == True):
                            #FinalSP[item.var_name[0]] = item
                        space.append(item)
                        #defaults.append(item.default)
                    #space.default = defaults
                    lsFinalSP.append(space)
                    del space
            else:
                for searchSpace in final:
                    #defaults = []
                    space=[]
                    for group in searchSpace:
                        for item in group:
                            #if (item.iskeep == True):
                                #FinalSP[item.var_name[0]] = item
                            space.append(item)
                            #defaults.append(item.default)
                    #space.default=defaults
                    lsFinalSP.append(space)
                    del space
        elif(len(MixList)==1):
            final=list(MixList)
            for searchSpace in final:
                for group in searchSpace:
                    #defaults = []
                    space=[]
                    for item in group:
                        if (item.iskeep == True):
                            FinalSP[item.var_name] = item
                            space.append(item)
                            #defaults.append(item.default)
                    #space.default = defaults
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

    def Combine(self,  Conditional: ConditionalSpace = None, Forbidden: Forbidden = None,
                isBandit: bool=True,sp_cluster=0, ifAllSolution=True, random_seed= 0,min_sp=3,
                n_init_sp=None, max_eval=500, init_sample=10, sample_sp=10, init_ratio=0.5) -> List[HyperParameter]:
        if (Conditional == None):
            isBandit=False
        if(isBandit == True):
            if ifAllSolution:
                return self._conditionalfree(Conditional,Forbidden,sp_cluster, ifAllSolution)
            else:
                return self._combinewithconditional(Conditional,Forbidden,sp_cluster, ifAllSolution,
                                                    random_seed,min_sp, n_init_sp, max_eval, init_sample,sample_sp, init_ratio=init_ratio)
        else:
            self._listconditional = Conditional
            self._listForbidden = Forbidden
            ''' 15/03/2021
            defaults=[]
            for _,item in self._hyperparameters.items():
                if 'space' not in locals():
                    space = item
                else:
                    space = space + item
                defaults.append(item.default)
            space.default = defaults
            return space'''
            lsSpace=[]
            for k,v in self._hyperparameters.items():
                lsSpace.append(v)
            return self
    def _combinewithconditional(self, cons: ConditionalSpace = None, forb: Forbidden = None,
                               sp_cluster=0, ifAllSolution=True, random_seed= 0,min_sp=3,
                                n_init_sp=None, max_eval=500, init_sample=10, sample_sp=10, init_ratio=0.5) -> List[HyperParameter]:
        _defratio=init_ratio
        np.random.seed(random_seed)
        _max_sp = int(np.floor(max_eval * _defratio) / sample_sp)
        _min_sp = min_sp if min_sp!=None else int(np.floor(init_sample/sample_sp))
        if _max_sp < _min_sp:
            _temp_sp = _min_sp
            _min_sp = _max_sp
            _max_sp = _temp_sp
        _max_sp=_max_sp+1 if _max_sp==_min_sp else _max_sp
        self._listconditional=cons
        self._listForbidden=forb
        lsParentName,lsChildEffect, lsFinalSP,lsVarNameinCons,childList  = [],[],[],[],[]
        listParam={i:k for i,k in self._hyperparameters.items()}
        for i, con in cons.conditional.items():
            if con[0] not in listParam.keys():
                raise TypeError("Hyperparameter '%s' is not exists in current ConfigSpace" %
                                str(con[0]))
            else:
                if(all(i in [j for i in [x.bounds for x in listParam[con[1]].bounds] for j in i] for i in con[2])==False):
                    raise TypeError("Value  '%s' doesnt exists" %
                                    str(con[2]))
            if con[1] not in listParam.keys():
                raise TypeError("Hyperparameter '%s' is not exists in current ConfigSpace" %
                                str(con[1]))
            if ([con[1], con[2]] not in lsParentName):
                for x in con[2]:
                    lsParentName.append([con[1], [x]])
            if(con[1] not in lsVarNameinCons):
                lsVarNameinCons.append(con[1])
            if con[0] not in childList:
                childList.append(con[0])

        lsOneNode = [x for x in listParam.keys() if x not in (lsVarNameinCons + childList)]
        #lsChildEffect.append([str(con[1]) + "_" + "".join(con[2]), con[0]])
        ##Check if child belongs to 2 parents (conflict case):
        #{i[0]: [x[0] for x in lsParentName].count(i[0]) for i in lsParentName}
        #for item in [item for item, count in collections.Counter([x[0] for x in lsParentName]).items() if count > 1]:

        #lsParentName = [t for t in (set(tuple(i) for i in lsParentName))]
        #lsVarNameinCons = np.unique(np.array(lsVarNameinCons))
        ##List out the branches which have values with no conditional
        _temp=[]

        _test=dict()
        _test2=dict()
        for vName in lsVarNameinCons+lsOneNode:
            # item_noCons = [x for x in self._hyperparameters[vName].allbounds if x not in [x[1] for x in lsParentName if x[0] ==vName]]
            #If the current node is algorithmChoice, listing by bounds, otherwise list based on conditional
            #if isinstance(self._hyperparameters[vName],AlgorithmChoice):
            if isinstance(self._hyperparameters[vName],(CategoricalParam,AlgorithmChoice)):
                #lsParentName=[x for x in lsParentName if x[0]!=vName]
                _thisNode=[]
                if len(self._hyperparameters[vName].bounds) > 1:
                    #for x in self._hyperparameters[vName].bounds:
                    _thisNode+=[x.bounds for x in self._hyperparameters[vName].bounds]
                    _test2[vName] = _thisNode
                else:
                    #for x in self._hyperparameters[vName].bounds[0].bounds:
                    _thisNode+=[x for x in self._hyperparameters[vName].bounds[0].bounds]
                    _test[vName]=_thisNode
            else:
                _thisNode = [x.bounds for x in self._hyperparameters[vName].bounds]
                _test2[vName] = _thisNode
        _allBounds = dict()
        for i, x in _test2.items():
            _allBounds[i] = [j for i in x for j in i]
        for i, x in _test.items():
            _allBounds[i] = x
        _a={i:[*range(1,len(v)+1)] if i in lsVarNameinCons else [1] for i,v in _test.items()}
        _i = []
        _lsParentName = []
        _tmax_sp = n_init_sp if n_init_sp!=None else _max_sp
        for i, x in _test.items():
            _i.append(len(x))
        for i,x in _test2.items():
            if isinstance(self._hyperparameters[i],(AlgorithmChoice)):
                _a[i] = [*range(1, len(x) + 1)]
            else:
                _a[i]=[1]
            _i.append(len([j for i in x for j in i]))
            #_a[i]=[len(x)]
            #_i.append(len(x))

        _ = [x for x in itertools.product(*list(_a.values())) if
                          np.product(x) in [*range(_min_sp,_tmax_sp+1)]]
        _splitStrategy=[]
        if len(_)>1:
            _splitStrategy = [x for x in itertools.product(*list(_a.values())) if
                              np.product(x) == _tmax_sp]
            while len(_splitStrategy) < 1:
                _tmax_sp = _tmax_sp - 1
                _splitStrategy = [x for x in itertools.product(*list(_a.values())) if
                                  np.product(x) == _tmax_sp]
        if len(_splitStrategy)<1:
            _a={i:[*range(1,len(v)+1)] for i,v in _test.items()}
            _lsParentName = []
            for i,x in _test2.items():
                _a[i] = [*range(1, len(x) + 1)]
                #_a[i]=[len(x)]
                #_i.append(len(x))
            _min_required=np.product(_i) if len(_i)>0 else 1
            if _min_required>_max_sp:
                raise TypeError("Not enought budget for '%s' (at least) sub-search spaces" %
                                _min_required)
            _splitStrategy=[]
            #if(n_init_sp!=None):
            _splitStrategy=[x for x in itertools.product(*list(_a.values())) if
                            np.product(x) ==(n_init_sp if n_init_sp!=None else _max_sp)]

            while len(_splitStrategy)<1:
                _tmax_sp=_tmax_sp-1
                _splitStrategy = [x for x in itertools.product(*list(_a.values())) if
                                  np.product(x) == _tmax_sp]
        #if (len(_splitStrategy)<1):
        #    _splitStrategy = [x for x in itertools.product(*list(_a.values())) if
         #                     np.product(x) in range(_min_sp, _max_sp)]
        if (len(_splitStrategy) < 1):
            raise TypeError("No spliting solution")
        '''_t=[]
        for x in _splitStrategy:
            _t.append(sum([1-i/j for i, j in zip(list(x), [x[-1] for x in _a.values()])]))
        '''
        _tarr, _ibest, _ibestValue = [], [], 0.00
        for i, x in enumerate(_splitStrategy):
            _pArr = 0.00
            for _, _x in enumerate(x):
                _nBounds = _i[_]
                _pro = 1 - (_x / _nBounds)
                _pArr = _pArr + _pro
            if _ibestValue == _pArr:
                _ibest.append(i)
            elif _ibestValue < _pArr:
                _ibestValue = _pArr
                _ibest = [i]
            else:
                pass
            _tarr.append(_pArr)
        _param_ori = []
        _splitStrategy = [_splitStrategy[i] for i in _ibest]
        if len(_splitStrategy) > 1:
            _tarr = [_tarr[i] for i in _ibest]
            _t = np.sum(_tarr)
            _p = [math.floor((x / _t) * 1000) / 1000 for x in _tarr[:-1]]
            _p.append(1 - sum(_p))
            _choosenstr = _splitStrategy[np.random.choice(len(_splitStrategy), p=_p)]
        else:
            _choosenstr = _splitStrategy[0]
        #new P:
        '''_tSum=np.sum(_t)
        _size = len(_splitStrategy)
        if len(_t)>1:
            _p=[math.floor(x*100/_tSum)/100 for x in _t[:-1]]
            _p.append(1-sum(_p))
            _r_choice = np.random.choice(_size, p=_p)
        else:
            _r_choice = 0
        _t
        _t=sum([np.product(x) for x in _splitStrategy])
        #np.random.seed(random_seed)
        _p=[(math.floor(np.product(x)*1000/_t))/1000 for x in _splitStrategy]'''

        #_choosenstr = _splitStrategy[_r_choice]
        for i,x in enumerate(_choosenstr):
            _key=list(_a)[i]
            _thisnode=_test[_key] if _key in _test else _test2[_key]
            if len(_thisnode)==x:
                for _item in _thisnode:
                    _lsParentName.append([_key,[_item]])#[[_key,z] for z in _thisnode]
            elif x==1:
                _groupvalue = []
                for _x in _thisnode:
                    if isinstance(_x, list):
                        _groupvalue.extend(_x)
                    else:
                        _groupvalue.append(_x)
                _lsParentName.append([_key,_groupvalue])
                #_lsParentName.append([_key,[j for i in [x for x in _thisnode] for j in i]])
            else:
                _x=x
                #_temp=[]
                #_thisnode=[j for i in [x for x in _thisnode] for j in i]
                _itemthisnode = {i: x for i, x in enumerate(_thisnode)}
                while len(_itemthisnode)>0:
                    _size = int(np.ceil(len(_itemthisnode) / _x))
                    _atemp = {i: (1 / (len(x) / len([j for i in list(_itemthisnode.values()) for j in i]))) for i, x in
                              _itemthisnode.items()}
                    _asum = sum(_atemp.values())
                    _ai = 0
                    for _ia, _xa in _atemp.items():
                        _ai = _ai + 1
                        if _ai >= len(_itemthisnode):
                            _atemp[_ia] = 1 - (sum(_atemp.values()) - _atemp[_ia])
                        else:
                            _atemp[_ia] = _xa / _asum
                    #np.random.seed(random_seed)
                    _group= list(np.random.choice(list(_itemthisnode.keys()),_size,replace=False, p=list(_atemp.values()))) \
                        if len(_itemthisnode)>_size else list(_itemthisnode.keys())
                    #_thisnode=list(set(_thisnode)-set(_group))
                    #_temp.append(list(_group))
                    _thisgroup = [x for idx, x in _itemthisnode.items() if idx in _group]
                    _itemthisnode = {i: x for i, x in _itemthisnode.items() if i not in _group}
                    _groupvalue=[]
                    for x in _thisgroup:
                        if isinstance(x,list):
                            _groupvalue.extend(x)
                        else:
                            _groupvalue.append(x)
                    _lsParentName.append([_key,_groupvalue])
                    _x = _x - 1

            #np.random.choice[_test[_key],]
        newlsParentName = []
        for item,count in collections.Counter([x[0] for x in _lsParentName]).items():
            if(count==1):
                newlsParentName.extend([[x[0],x[1]] for x in _lsParentName if x[0]==item])
            else:
                temp = [[x[0], len(x[1]), x[1]] for x in _lsParentName if x[0] == item]
                temp.sort(reverse=False)
                feeded = []
                for index, rootvalue in enumerate(temp):
                    # print(index,rootvalue)
                    flag = False
                    for value in temp[index + 1:]:
                        abc = set(rootvalue[2]).intersection(set(value[2]))
                        if (len(abc) > 0):
                            flag = True
                        if (len(set(rootvalue[2]).intersection([item for sublist in feeded for item in sublist])) > 0):
                            flag = True
                    if (flag == True):
                        for i in rootvalue[2]:
                            if (i not in ([item for sublist in feeded for item in sublist])):
                                newlsParentName.append([rootvalue[0], [i]])
                                feeded.append([i])
                    else:
                        dif = list(set(rootvalue[2]).difference([item for sublist in feeded for item in sublist]))
                        newlsParentName.append([rootvalue[0], dif])
                        feeded.append(dif)
        for item in newlsParentName:
            #con=
            _thisnode=item[1]#[j for i in [x for x in item[1]] for j in i]
            for con in [x for x in cons.conditional.values() if x[1]==item[0] and len(set(x[2]).intersection(set(_thisnode)))]:
                lsChildEffect.append([str(con[1]) + "_" + "".join(_thisnode), con[0]])
        lsSearchSpace = self.listoutAllBranches(lsVarNameinCons, lsChildEffect, newlsParentName,sp_cluster)
        return lsSearchSpace
    #def _checkForbidden(self, lsSearchSpace):
    def _conditionalfree(self, cons: ConditionalSpace = None, forb: Forbidden = None,
                               sp_cluster=0, ifAllSolution=True) -> List[HyperParameter]:
        self._listconditional=cons
        self._listForbidden=forb
        listParam = OrderedDict()
        ordNo = 0
        lsParentName,lsChildEffect, lsFinalSP,lsVarNameinCons  = [],[],[],[]
        for i, param in self._hyperparameters.items():
            #lsVarName.append(i)
            listParam[i] = param.allbounds
            '''if len(param.id_N) >= 1:
                self._OrgLevels[ordNo] = param.bounds[0]
                ordNo += 1
        self.dim = ordNo'''
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
            #lsChildEffect.append([str(con[1]) + "_" + "".join(con[2]), con[0]])
        ##Check if child belongs to 2 parents (conflict case):
        #{i[0]: [x[0] for x in lsParentName].count(i[0]) for i in lsParentName}
        #for item in [item for item, count in collections.Counter([x[0] for x in lsParentName]).items() if count > 1]:

        #lsParentName = [t for t in (set(tuple(i) for i in lsParentName))]
        #lsVarNameinCons = np.unique(np.array(lsVarNameinCons))
        ##List out the branches which have values with no conditional
        if (ifAllSolution == True):
            for vName in lsVarNameinCons:
                #item_noCons = [x for x in self._hyperparameters[vName].allbounds if x not in [x[1] for x in lsParentName if x[0] ==vName]]
                itemValues = self._hyperparameters[vName].allbounds
                itemThisNode= [x[1] for x in lsParentName if x[0] ==vName]
                item_noCons=[]
                if (len(itemThisNode)>0):
                    itemThisNode2=[]
                    for item in itemThisNode:
                        itemThisNode2+=item
                    item_noCons = [x for x in itemValues if x not in itemThisNode2]

                if(len(item_noCons)>0):
                    item_noCons = item_noCons if len(item_noCons) < 2 else list(item_noCons)
                    lsParentName.append([vName, item_noCons])

        newlsParentName = []
        for item,count in collections.Counter([x[0] for x in lsParentName]).items():
            if(count==1):
                newlsParentName.extend([[x[0],x[1]] for x in lsParentName if x[0]==item])
            else:
                temp = [[x[0], len(x[1]), x[1]] for x in lsParentName if x[0] == item]
                temp.sort(reverse=False)
                feeded = []
                for index, rootvalue in enumerate(temp):
                    # print(index,rootvalue)
                    flag = False
                    for value in temp[index + 1:]:
                        abc = set(rootvalue[2]).intersection(set(value[2]))
                        if (len(abc) > 0):
                            flag = True
                        if (len(set(rootvalue[2]).intersection([item for sublist in feeded for item in sublist])) > 0):
                            flag = True
                    if (flag == True):
                        for i in rootvalue[2]:
                            if (i not in ([item for sublist in feeded for item in sublist])):
                                newlsParentName.append([rootvalue[0], [i]])
                                feeded.append([i])
                    else:
                        dif = list(set(rootvalue[2]).difference([item for sublist in feeded for item in sublist]))
                        newlsParentName.append([rootvalue[0], dif])
                        feeded.append(dif)
        for item in newlsParentName:
            #con=
            for con in [x for x in cons.conditional.values() if x[1]==item[0] and len(set(x[2]).intersection(item[1]))]:
                lsChildEffect.append([str(con[1]) + "_" + "".join(item[1]), con[0]])
        lsSearchSpace = self.listoutAllBranches(lsVarNameinCons, lsChildEffect, newlsParentName,sp_cluster)
        return lsSearchSpace

if __name__ == '__main__':
    np.random.seed(1)
    cs = ConfigSpace()
    alg_namestr = CategoricalParam([("SVM", 0.4), "RF", ['LR', 'DT']], "alg_namestr")
    test = CategoricalParam(("A", "B"), "test", default="A")
    testCD = CategoricalParam(("C", "D"), "testCD", default="C")
    C = FloatParam([1e-2, 100], "C")
    degree = IntegerParam([([1, 2], 0.1), ([3, 5], .44), [6, 10], 12], 'degree')
    f = FloatParam([(0.01, 0.5), [0.02, 100]], "testf")
    con = ConditionalSpace("test")
    #arange=range(1, 50, 2)
    abc = CategoricalParam([x for x in range(1, 50, 2)], "abc")
    cs.add_multiparameter([alg_namestr, test, C, degree, f,abc,testCD])
    con.addConditional(test, alg_namestr, "SVM")
    con.addMutilConditional([test,degree],alg_namestr,"RF")
    fobr = Forbidden()
    #fobr.addForbidden(abc, 5, alg_namestr, "SVM")
    fobr.addForbidden(test,'A',abc, 5)
    fobr.addForbidden(test, 'B', abc, 7)
    fobr.addForbidden(testCD, 'C', abc, 1)
    lsSpace = cs.Combine(con,fobr,isBandit=True)
    lsSpace
