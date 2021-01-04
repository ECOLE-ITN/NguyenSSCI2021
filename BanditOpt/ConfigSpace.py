from __future__ import print_function
from copy import deepcopy
from collections import OrderedDict
from typing import Union, List, Dict, Optional
from numpy.random import randint
import itertools, collections
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer
#import mipego4ml.ConditionalSpace
from BanditOpt.Forbidden import Forbidden
from BanditOpt.ConditionalSpace import ConditionalSpace
from BanditOpt.ParamExtension import rebuild
from sklearn.cluster import AgglomerativeClustering
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
    def _getnodechilds(self,node,childeffect,lsvarname,lsparentname):
        if (isinstance(self._hyperparameters[node[0]], NominalSpace)==False):
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
                    childlst=[child_hpa_i,list(self._hyperparameters[child_hpa_i].bounds[0])]
                    child_node.append(childlst)
                    #childlst=[x for x in self._listconditional.conditional.values() if
                                        #x[0] == child_hpa_i and x[2] == node[1]]

        return child_hpa,child_node
    def _listoutallnode(self, node, rootname, lsvarname, childeffect,
                               lsparentname, mixlist,feeded):
        temp = deepcopy(self._hyperparameters[node[0]])
        if(temp.var_name[0] not in feeded):
            feeded.append(temp.var_name[0])
        if (isinstance(node[1], tuple)):
            temp.bounds = [node[1]]
        else:
            temp.bounds = [tuple(node[1])]
        temp.iskeep = False
        temp = rebuild(temp)
        this_node = [temp]
        if (isinstance(temp, NominalSpace)):
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
            for item in [s for s in mixlist if s.var_name[0] == x]:
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
            rootnode = [x for x in sp if x.iskeep == False and x.var_name[0] == root][0]
            rootnode.iskeep=True
            while (len(childs)>0):
                childofChild=[]
                for child in childs:
                    item= [x for x in sp if x.iskeep==False and x.var_name[0]==child[0]][0]
                    childvalue=list(item.bounds[0])
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
                if (isinstance(i,NominalSpace)):
                    temp_id=temp_id+i.var_name[0]+"_"+"_".join(str(e) for e in list(i.bounds[0]))
                else:
                    temp_id = temp_id + i.var_name[0]
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
            temp = rebuild(temp)
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
            temp=rebuild(temp)
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
            temp= rebuild(temp)
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
            temp=rebuild(temp)
            #hpchild = [temp]
            hpi.append(temp)
        return hpi
    def _clustering(self,final,sp_cluster):

        header=[x.var_name[0] for x in self._hyperparameters.values() if isinstance(x,NominalSpace)]
        notcount=[x.var_name[0] for x in self._hyperparameters.values() if isinstance(x,NominalSpace)==False]
        le = LabelEncoder()
        LstEnc = dict()
        for i in header:
            item = self._hyperparameters[i]
            le.fit(item.bounds[0])
            LstEnc[i] = le.classes_
        df=[]
        for idx,sp in enumerate(final):
            itemarr= [idx]
            for item in header:
                try:
                    #i = sp[item]
                    i=[x for sublist in sp for x in sublist if x.var_name[0] == item][0]
                    ivalue=i.bounds[0]
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
                    if (x.var_name[0] in temp.keys()):
                        lastvalue=temp[x.var_name[0]].bounds[0]
                        thisvalue=x.bounds[0]
                        diff= tuple(set(thisvalue)-set(lastvalue))
                        if(len(diff)>0):
                            temp[x.var_name[0]].bounds[0]=lastvalue+diff
                    else:
                        temp[x.var_name[0]]=x
            for _,x in temp.items():
                x=rebuild(x)
            newFinal.append(temp)
        return newFinal
    def listoutAllBranches(self, lsvarname, childeffect, lsparentname,sp_cluster=0) -> List[SearchSpace]:
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
                finalA=self._listoutBranches4(item,root,lsvarname, childeffect, lsparentname)
                lsBranches.extend(finalA)
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
                            #print(index_right)
                        i+=1
                    if (item_left!=None and item_right!=None):
                        sp_bound_left = item_left.bounds[0]
                        sp_bound_left = tuple(set(sp_bound_left) - set(value.leftvalue))
                        sp_bound_right = item_right.bounds[0]
                        sp_bound_right_remain = tuple(set(sp_bound_right) - set(value.rightvalue))
                        if (len(sp_bound_right_remain) < 1 and len(sp_bound_left) < 1 and isBothRoot == True):
                            isDelete = True
                        if (len(sp_bound_left) > 0):
                            item_left.bounds[0]=sp_bound_left
                            item_left=rebuild(item_left)
                            group_new[module_left][index_left]=item_left
                        else:
                            if (isDelete==False):
                                left_childs=[ke[0] for ke in self._listconditional.conditional.values() if ke[1]==item_left.var_name[0]
                                             and len(set(item_left.bounds[0])-set(ke[2]))==0]
                                lIndex_del=[index_left]
                                lChild_del=[]
                                #group_new[module_left].pop(index_left)
                                while(len(left_childs)>0):
                                    for left_child in left_childs:
                                        left_childs.extend([ke[0] for ke in self._listconditional.conditional.values() if
                                         ke[1] == left_child])
                                        lChild_del.append(left_child)
                                        left_childs.remove(left_child)
                                lIndex_del.extend([idx for (idx, x) in enumerate(group_new[module_left]) if
                                                  x.var_name[0] in lChild_del])
                                for index in sorted(lIndex_del, reverse=True):
                                    del group_new[module_left][index]

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
                    for _,item in group.items():
                        #if (item.iskeep == True):
                            #FinalSP[item.var_name[0]] = item
                        if 'space' not in locals():
                            space = item
                        else:
                            space = space + item
                    lsFinalSP.append(space)
                    del space
            else:
                for searchSpace in final:
                    for group in searchSpace:
                        for item in group:
                            #if (item.iskeep == True):
                                #FinalSP[item.var_name[0]] = item
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
    def Combine(self,  Conditional: ConditionalSpace = None, Forbidden: Forbidden = None,isBandit: bool=True,sp_cluster=0, ifAllSolution=True) -> List[SearchSpace]:
        if (Conditional == None):
            isBandit=False
        if(isBandit == True):
            return self.combinewithconditional(Conditional,Forbidden,sp_cluster, ifAllSolution)
        else:
            self._listconditional = Conditional
            self._listForbidden = Forbidden
            for _,item in self._hyperparameters.items():
                if 'space' not in locals():
                    space = item
                else:
                    space = space + item
            return space
    def combinewithconditional(self, cons: ConditionalSpace = None, forb: Forbidden = None,sp_cluster=0, ifAllSolution=True) -> List[SearchSpace]:
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
            #lsChildEffect.append([str(con[1]) + "_" + "".join(con[2]), con[0]])
        ##Check if child belongs to 2 parents (conflict case):
        #{i[0]: [x[0] for x in lsParentName].count(i[0]) for i in lsParentName}
        #for item in [item for item, count in collections.Counter([x[0] for x in lsParentName]).items() if count > 1]:

        #lsParentName = [t for t in (set(tuple(i) for i in lsParentName))]
        #lsVarNameinCons = np.unique(np.array(lsVarNameinCons))
        ##List out the branches which have values with no conditional
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
                        lsParentName.append([vName, item_noCons])
                    else:
                        lsParentName.append([vName, list(item_noCons)])
                    #','.join([str(elem) for elem in noCon])
                #for a3 in noCon:
                #    lsParentName.append(tuple([a1, a3]))
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
    lsSpace = cs.combinewithconditional()
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