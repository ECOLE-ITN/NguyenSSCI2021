from Component.mHyperopt import hp
from BanditOpt.ConfigSpace import ConfigSpace, SearchSpace, ConditionalSpace, NominalSpace, ContinuousSpace, OrdinalSpace
import copy
import numpy as np

def SubToHyperopt(search_space: list, con: ConditionalSpace, prefix='value'):
    Ls_hpOpt = []
    for ls in search_space:
        if con!=None:
            condx= [x for _, x in con.AllConditional.items() if x[0] in ls.var_name and x[1] in ls.var_name]
        else:
            condx=[]
        if(len(condx)<1):
            hOpt = dict()
            varname = ls.var_name
            levels = list(ls.levels.items())
            bounds = ls.bounds
            vartype = ls.var_type
            i = 0
            while i < len(varname):
                if (i in ls.id_N):  # Categorical
                    hOpt[varname[i]] = hp.choice(varname[i], (([a for a in bounds[i]])))
                elif (i in ls.id_C):  ##real number
                    hOpt[varname[i]] = hp.uniform(varname[i], float(bounds[i][0]), float(bounds[i][1]))
                elif (i in ls.id_O):  ##interge
                    hOpt[varname[i]] = hp.choice(varname[i], range(int(bounds[i][0]), int(bounds[i][1])))
                i = i + 1
            # print('lstparams.append(cs)')
            Ls_hpOpt.append(copy.deepcopy(hOpt))
    #Ls_hpOpt = []
    #for ls in search_space:
        else:
            lsParentName, childList, lsFinalSP, ActiveLst, noCheckForb = [], [], [], [], []
            sp = dict(zip(ls.var_name, ls.bounds))
            var_names = ls.var_name
            lsParents = []
            for c in [x for _, x in con.AllConditional.items() if x[0] in sp.keys() and x[1] in sp.keys()]:
                if ([[c[1], c[2]], c[0]] not in lsParentName):
                    childvalue = sp[c[0]]
                    lsParentName.append([[c[1], tuple(c[2])], c[0]])
            lsRootNode = list(np.unique(np.array([xp[0] for xp, xc in lsParentName])))
            lsChildNode = [x for _, x in lsParentName]
            lsOneNode = [x for x in sp if x not in (lsRootNode + lsChildNode)]
            if (len(lsOneNode) > 0):
                for x in lsOneNode:
                    itemValues = sp[x]
                    lsParentName.append([[x, itemValues], None])
            for vName in lsRootNode:
                itemValues = sp[vName]
                itemThisNode = [x[1] for x, _ in lsParentName if x[0] == vName]
                item_noCons = []
                if (len(itemThisNode) > 0):
                    itemThisNode2 = []
                    for item in itemThisNode:
                        itemThisNode2 += item
                    item_noCons = [x for x in itemValues if x not in itemThisNode2]
                    #print(itemThisNode2)
                if (len(item_noCons) > 0):
                    if (len(item_noCons) < 2):
                        # print('+++', vName)
                        lsParentName.append([[vName, tuple(item_noCons)], None])
                    else:
                        # print('---', vName)
                        lsParentName.append([[vName, tuple(item_noCons)], None])
            #print(lsParentName)
            finaldict = dict()
            lsr = []
            for x in [x for x, _ in lsParentName if x[0] not in [x for _, x in lsParentName]]:
                if (x not in lsr):
                    lsr.append(x)
                    xxx = _xxxsingle(x[0], x[0], x[1], None,prefix, lsParentName, sp, con)
                    if (x[0] not in finaldict.keys()):
                        finaldict.update(xxx)
                    else:
                        # print(xxx)
                        finaldict[x[0]].append(xxx[x[0]][0])
            hOpt = _formatsingle(finaldict, None, None, sp, ls,prefix)
            if(len(hOpt)>1):
                hOpt_joint= hp.choice('rootxxxx', [hOpt])
            else:
                hOpt_joint=hOpt
            Ls_hpOpt.append(copy.deepcopy(hOpt_joint))
    return Ls_hpOpt
'''def ToHyperopt(search_space: SearchSpace, AllConditional: dict(),BreakConditional:dict()):
    lsParentName, childList, lsFinalSP, ActiveLst, noCheckForb = [], [], [], [], []
    for i, con in AllConditional.items():
        #0:child, 1: parent, 2: parentValue
        if ([con[1], con[2], con[0]] not in lsParentName):
            lsParentName.append([con[1], con[2], con[0]])
        if (con[0] not in childList):
            childList.append(con[0])
    lsRootNode = [x for x in lsParentName[1] if x not in childList]
    for root in lsRootNode:
        root'''
def OrginalToHyperopt(search_space: SearchSpace, con: ConditionalSpace, prefix='value'):
    if con != None:
        condx = [x for _, x in con.AllConditional.items()]
    else:
        condx = []
    if (len(condx) < 1):
        hOpt=dict()
        ls=search_space._hyperparameters
        for varname,x in ls.items():
            if (isinstance(x,NominalSpace)):  # Categorical
                hOpt[varname] = hp.choice(varname,x.bounds[0])
            elif (isinstance(x,ContinuousSpace)):  ##real number
                hOpt[varname] = hp.uniform(varname, float(x.bounds[0][0]), float(x.bounds[0][1]))
            elif (isinstance(x,OrdinalSpace)):  ##interge
                hOpt[varname] = hp.choice(varname, range(int(x.bounds[0][0]), int(x.bounds[0][1])))

        # print('lstparams.append(cs)')
        jointsp=hOpt
    else:
        lsParentName, childList, lsFinalSP, ActiveLst, noCheckForb = [], [], [], [], []
        sp = search_space._hyperparameters
        var_names = [x for x, _ in search_space._hyperparameter_idx.items()]
        lsParents = []
        for i, c in con.AllConditional.items():
            if ([[c[1], c[2]], c[0]] not in lsParentName):
                childvalue = sp[c[0]].bounds
                lsParentName.append([[c[1], tuple(c[2])], c[0]])
        lsRootNode = list(np.unique(np.array([xp[0] for xp, xc in lsParentName])))
        lsChildNode = [x for _, x in lsParentName]
        lsOneNode = [x for x in sp if x not in (lsRootNode + lsChildNode)]
        if (len(lsOneNode) > 0):
            for x in lsOneNode:
                # print(x)
                itemValues = sp[x].bounds[0]
                lsParentName.append([[x, itemValues], None])
        for vName in lsRootNode:
            itemValues = sp[vName].bounds[0]
            #print(vName, itemValues)
            itemThisNode = [x[1] for x, _ in lsParentName if x[0] == vName]
            item_noCons = []
            if (len(itemThisNode) > 0):
                itemThisNode2 = []
                for item in itemThisNode:
                    itemThisNode2 += item
                item_noCons = [x for x in itemValues if x not in itemThisNode2]
                #print(itemThisNode2)
            if (len(item_noCons) > 0):
                if (len(item_noCons) < 2):
                    #print('+++', vName)
                    lsParentName.append([[vName, tuple(item_noCons)], None])
                else:
                    #print('---', vName)
                    lsParentName.append([[vName, tuple(item_noCons)], None])
        finaldict = dict()
        lsr = []
        for x in [x for x, _ in lsParentName if x[0] not in [x for _, x in lsParentName]]:
            if (x not in lsr):
                #print(x)
                lsr.append(x)
                xxx = _xxx(x[0], x[0], x[1], None, lsParentName, sp,prefix)
                if (x[0] not in finaldict.keys()):
                    finaldict.update(xxx)
                else:
                    # print(xxx)
                    finaldict[x[0]].append(xxx[x[0]][0])
        finasp=_format(finaldict,None,None,sp,prefix)
        if(len(finasp)>1):
            jointsp = hp.choice('rootxxxx', [finasp])
        else:
            jointsp=finasp
    return jointsp
def _xxx(rootname, node, value, parent, lsParentName,sp,prefix):
    child_hpa, child_node = _getchilds(node, value, lsParentName,sp)
    thisnode=dict()
    isExist=False
    if(len(child_hpa)>0):
        fnode=node
        thisnode[node]=[{prefix:value}]
        for child in child_node:
            hpa, _ = _getchilds(child[0],child[1], lsParentName,sp)
            childnode=dict()
            if(len(hpa)>0):
                childnode=_xxx(rootname, child[0],child[1],node, lsParentName,sp,prefix)
            else:
                if(child[0] in thisnode[node][0].keys()):
                    childnode[prefix]=child[1]
                    isExist=True
                else:
                    childnode[child[0]]=child[1]
            if (child[0] in thisnode[node][0].keys()):
                if(list(childnode.keys())[0]!=prefix):
                    first_value = next(iter(childnode.values()))
                    thisnode[node][0][child[0]].append(first_value[0])
                else:
                    thisnode[node][0][child[0]].append(childnode)
            else:
                thisnode[node][0].update(childnode)
    else:
        if(len([x for x,_ in lsParentName if x[0]==node])>1):
            thisnode[node]=[{prefix:value}]
        else:
            thisnode[node]=value
    return thisnode
def _getchilds(node, value,lsParentName,sp):
    child_node=[]
    child_hpa=[_ for x,_ in lsParentName if x[0]==node and x[1]==value and _!=None]
    for child in child_hpa:
        #print(child,[x for x,_ in lsParentName if x[0]==child])
        if (len([x for x,_ in lsParentName if x[0]==child])>0):
            thisChild= [x for x,_ in lsParentName if x[0] == child]
            child_node.extend(thisChild)
        else:
            #print(child)
            item=sp[child].bounds[0]
            thisChild=[child,item]
        #print(thisChild)
            child_node.append(thisChild)
    return child_hpa, child_node
def _format(node,alg,parent, sp, prefix):
    fnode=node
    thisnode=dict()
    #print(node,alg)
    if (isinstance(fnode, list)):
        childlst=[]
        for x in fnode:
            child=_format(x,alg,alg, sp,prefix)
            childlst.append(child)
        thisnode[alg]=hp.choice(alg,childlst)
    elif(isinstance(fnode,dict)):
        childlst=dict()
        for i,v in fnode.items():
            #print('==>',i,v)
            child=_format(v,i,alg, sp,prefix)
            childlst.update(child)
        return childlst
    else:
        bkalg = alg
        bkparam = alg
        if (alg == prefix):
            bkparam = alg + "".join(node)
            alg = parent
        if (isinstance(sp[alg],NominalSpace)):
            thisnode[bkalg]=hp.choice(bkparam,node)
        elif(isinstance(sp[alg],ContinuousSpace)):
            thisnode[bkalg]=hp.uniform(bkparam,float(node[0]),float(node[1]))
        elif(isinstance(sp[alg],OrdinalSpace)):
            thisnode[bkalg]=hp.choice(bkparam, range(int(node[0]),int(node[1])))
    return thisnode
def _xxxsingle(rootname, node, value, parent, prefix, lsParentName,sp, con):
    child_hpa, child_node = _getchilds_single(node, value, lsParentName,sp)
    thisnode=dict()
    isExist=False
    if(len(child_hpa)>0):
        fnode=node
        thisnode[node]=[{prefix:value}]
        for child in child_node:
            hpa, _ = _getchilds_single(child[0],child[1], lsParentName,sp)
            childnode=dict()
            if(len(hpa)>0):
                childnode=_xxxsingle(rootname, child[0],child[1],node,prefix,
                                     lsParentName,sp, con)
            else:
                if(child[0] in thisnode[node][0].keys()):
                    childnode[prefix]=child[1]
                    isExist=True
                else:
                    childnode[child[0]]=child[1]
            if (child[0] in thisnode[node][0].keys()):
                if(list(childnode.keys())[0]!=prefix):
                    first_value = next(iter(childnode.values()))
                    thisnode[node][0][child[0]].append(first_value[0])
                else:
                    thisnode[node][0][child[0]].append(childnode)
            else:
                thisnode[node][0].update(childnode)
    else:
        if(len([x for x,_ in lsParentName if x[0]==node])>1):
            thisnode[node]=[{prefix:value}]
        else:
            if(len([x[2] for _,x in con.conditional.items() if x[1]==node])>0):
                thisnode[node] = [{prefix: value}]
            else:
                thisnode[node]=value
    return thisnode
def _getchilds_single(node, value,lsParentName,sp):
    child_node=[]
    child_hpa=[_ for x,_ in lsParentName if x[0]==node and x[1]==value and _!=None]
    for child in child_hpa:
        #print(child,[x for x,_ in lsParentName if x[0]==child])
        if (len([x for x,_ in lsParentName if x[0]==child])>0):
            thisChild= [x for x,_ in lsParentName if x[0] == child]
            child_node.extend(thisChild)
        else:
            #print(child)
            item=sp[child]
            thisChild=[child,item]
        #print(thisChild)
            child_node.append(thisChild)
    return child_hpa, child_node
def _formatsingle(node,alg,parent, sp,productspace, prefix='value'):
    fnode=node
    thisnode=dict()
    #print(node,alg)
    if (isinstance(fnode, list)):
        childlst=[]
        for x in fnode:
            child=_formatsingle(x,alg,alg, sp,productspace,prefix)
            childlst.append(child)
        thisnode[alg]=hp.choice(alg,childlst)
    elif(isinstance(fnode,dict)):
        childlst=dict()
        for i,v in fnode.items():
            #print('==>',i,v)
            child=_formatsingle(v,i,alg, sp,productspace,prefix)
            childlst.update(child)
        return childlst
    else:
        bkalg = alg
        bkparam = alg
        if (alg == prefix):
            bkparam = alg + "".join(node)
            alg = parent
        idx = productspace.var_name.index(alg)
        typex=''
        if (idx in productspace.id_C):
            typex='C'
        elif(idx in productspace.id_N):
            typex='N'
        elif(idx in productspace.id_O):
            typex='O'
        if (typex=='N'):
            thisnode[bkalg]=hp.choice(bkparam,node)
        elif(typex=='C'):
            thisnode[bkalg]=hp.uniform(bkparam,float(node[0]),float(node[1]))
        elif(typex=='O'):
            thisnode[bkalg]=hp.choice(bkparam, range(int(node[0]),int(node[1])))
    return thisnode
if __name__ == '__main__':
    from hyperopt import hp, tpe, rand, Trials
    import numpy as np
    randomstate=12
    Hspace = hp.choice('classifier_type', [
        {
            'type': hp.choice('type', ['naive_bayes', 'A', 'B']),
        },
        {
            'type': 'svm',
            'C': hp.lognormal('svm_C', 0, 1),
            'kernel': hp.choice('svm_kernel', [
                {'type': 'linear'},
                {'type': 'RBF', 'width': hp.lognormal('svm_rbf_width', 0, 1)},
            ]),
        },
        {
            'type': 'dtree',
            'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
            'max_depth': hp.choice('dtree_max_depth',
                                   [None, hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
            'min_samples_split': hp.qlognormal('dtree_min_samples_split', 2, 1, 1),
        },
    ])

    from hyperopt import fmin, tpe


    def new_obj(params):
        print(params)
        return (np.random.uniform(0, 1))


    best_candidate = fmin(new_obj, Hspace, algo=tpe.suggest, max_queue_len=4, max_evals=20)