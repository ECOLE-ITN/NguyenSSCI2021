from Component.mHyperopt import hp
#from BanditOpt.ConfigSpace import ConfigSpace, SearchSpace, ConditionalSpace, NominalSpace, ContinuousSpace, OrdinalSpace
import copy, math
from BanditOpt.ConfigSpace import ConfigSpace,ConditionalSpace, Forbidden,HyperParameter, FloatParam, CategoricalParam, IntegerParam, AlgorithmChoice
import numpy as np
from BanditOpt.ParamRange import p_paramrange,paramrange,one_paramrange

def SubToHyperopt(search_space: list, con: ConditionalSpace, prefix='value'):
    Ls_hpOpt = []
    for ls in search_space:
        _var_names= [x.var_name for x in ls]
        if con!=None:
            condx= [x for _, x in con.AllConditional.items() if x[0] in _var_names and x[1] in _var_names]
        else:
            condx=[]
        if(len(condx)<1):
            #No conditional
            hOpt = dict()
            for x in ls:
                hOpt[x.var_name]=_toHyperoptSyntax(x,x.var_name,x.var_name,prefix)
            Ls_hpOpt.append(copy.deepcopy(hOpt))
    #Ls_hpOpt = []
    #for ls in search_space:
        else:
            lsParentName, childList, lsFinalSP, ActiveLst, noCheckForb = [], [], [], [], []
            _allbounds = [x.bounds for x in ls]
            sp=dict((x.var_name, x) for x in ls)
            #sp = dict(zip(_var_names, _allbounds))
            _Parent_checked=[]
            for c in condx:
                _pName, _pValues, _cName = c[1], c[2], c[0]
                if len(set(_pValues).intersection(sp[_pName].allbounds))<1:
                    continue
                if ([_pName, _pValues, _cName] not in _Parent_checked):
                    _pbounds2=[x for x in lsParentName if x[0][0]==_pName and len(set(_pValues).intersection(x[0][1]))>0]
                    _pbounds3 = [x for x in _Parent_checked if
                                 x[0] == _pName and len(set(_pValues).intersection(x[1])) > 0]
                    if len(_pbounds2)>0:
                        _existBounds=list(np.unique([j for i in [x[1] for x in _pbounds3] for j in i]))
                        lsParentName=[x for x in lsParentName if x not in _pbounds2]
                        _Parent_checked = [x for x in _Parent_checked if x not in _pbounds3]
                    #_pParam=copy.deepcopy(sp[_pName])
                    for _x in _pbounds2:
                        _oldDiff, _newDiff, _samelst = [], [], []
                        _pParam=None
                        #_pbounds2.remove(x)
                        _olditem, _oldchild = _x[0], _x[1]
                        _oldDiff+=list(set(_olditem[1])-set(_pValues))
                        _copieditem=[x for x in _pbounds2 if x[0][1]==_oldDiff]
                        if len(_oldDiff)>0 and len(_copieditem)>0:
                            if [_pName, _oldDiff, _oldchild] not in _Parent_checked:
                                _item = copy.deepcopy(_copieditem[0])
                                _item[1] = _oldchild
                                lsParentName.append(_item)
                                _Parent_checked.append([_pName, _oldDiff, _oldchild])
                        elif len(_oldDiff)>0 and [_pName, _oldDiff, _oldchild] not in _Parent_checked:
                            '''_pParam = copy.deepcopy(sp[_pName])
                            _pbounds = [x for x in _pParam.bounds if len(set(_oldDiff).intersection(x.bounds)) > 0]
                            for x in _pbounds:
                                x.bounds = list(set(_oldDiff).intersection(x.bounds))
                            _pParam.bounds = _pbounds
                            _pParam.allbounds = _oldDiff'''
                            lsParentName.append([[_pName, _oldDiff, None], _oldchild])
                            _Parent_checked.append([_pName, _oldDiff, _oldchild])
                        try:
                            _newDiff+=list(set(_pValues)-set(_existBounds))
                        except:
                            pass
                        _copieditem = [x for x in _pbounds2 if x[0][1] == _newDiff]
                        if len(_newDiff)>0 and len(_copieditem) > 0:
                            _item = copy.deepcopy(_copieditem[0])
                            _item[1] =_cName
                            lsParentName.append(_item)
                            _Parent_checked.append([_pName, _newDiff, _cName])
                        elif len(_newDiff)>0 and [_pName, _newDiff, _cName] not in _Parent_checked:
                            '''_pParam = copy.deepcopy(sp[_pName])
                            _pbounds = [x for x in _pParam.bounds if len(set(_newDiff).intersection(x.bounds)) > 0]
                            for x in _pbounds:
                                x.bounds = list(set(_newDiff).intersection(x.bounds))
                            _pParam.bounds = _pbounds
                            _pParam.allbounds = _newDiff'''
                            lsParentName.append([[_pName, _newDiff, None], _cName])
                            _Parent_checked.append([_pName, _newDiff, _cName])

                        _samelst+=sorted(list(set(_pValues).intersection(_olditem[1])))
                        _copieditem = [x for x in _pbounds2 if x[0][1] == _samelst]
                        if len(_samelst)>0and len(_copieditem) > 0:

                            if [_pName, _samelst, _cName] not  in _Parent_checked:
                                _item = copy.deepcopy(_copieditem[0])
                                _item[1] = _cName
                                lsParentName.append(_item)
                                _Parent_checked.append([_pName, _samelst, _cName])

                            if [_pName, _samelst, _oldchild] not in _Parent_checked:
                                _item = copy.deepcopy(_copieditem[0])
                                _item[1] = _oldchild
                                lsParentName.append(_item)
                                _Parent_checked.append([_pName, _samelst, _oldchild])
                        elif len(_samelst)>0:
                            #pass
                            '''_pParam = copy.deepcopy(sp[_pName])
                            _pbounds = [x for x in _pParam.bounds if len(set(_samelst).intersection(x.bounds)) > 0]
                            for x in _pbounds:
                                x.bounds = list(set(_samelst).intersection(x.bounds))
                            _pParam.bounds = _pbounds
                            _pParam.allbounds = _samelst'''
                            _item = [_pName, _samelst, _cName]
                            if _item not in _Parent_checked:
                                lsParentName.append([[_pName, _samelst, None], _cName])
                                _Parent_checked.append([_pName, _samelst, _cName])
                            _item = [_pName, _samelst, _oldchild]
                            if _item not in _Parent_checked:
                                lsParentName.append([[_pName, _samelst, None], _oldchild])
                                _Parent_checked.append([_pName, _samelst, _oldchild])
                    #if len(_samelst)>0:

                #if ([_pName, _pValues, _cName] not in _Parent_checked):
                    if len(_pbounds2)==0:
                        '''_pParam = copy.deepcopy(sp[_pName])
                        _pbounds = [x for x in _pParam.bounds if len(set(_pValues).intersection(x.bounds)) > 0]
                        for x in _pbounds:
                            x.bounds=list(set(_pValues).intersection(x.bounds))
                        _pParam.bounds=_pbounds
                        _pParam.allbounds=_pValues'''
                        lsParentName.append([[_pName,_pValues, None], _cName])
                        _Parent_checked.append([_pName, _pValues, _cName])
            lsRootNode = list(np.unique(np.array([xp[0] for xp, xc in lsParentName])))
            lsChildNode = [x for _, x in lsParentName]
            lsOneNode = [x for x in sp if x not in (lsRootNode + lsChildNode)]
            if (len(lsOneNode) > 0):
                for x in lsOneNode:
                    itemValues = sp[x]
                    _Parent_checked.append([x, itemValues.allbounds, None])
                    lsParentName.append([[x, itemValues.allbounds,None], None])
            for vName in lsRootNode:
                itemValues = sp[vName]
                #itemThisNode = [x[1]for x in _Parent_checked if x[0] == vName]
                itemThisNode =[j for i in [x[1] for x in _Parent_checked if x[0] == vName] for j in i]
                item_noCons = []
                if (len(itemThisNode) > 0):
                    itemThisNode2 = np.unique(itemThisNode)
                    #for item in itemThisNode:
                     #   itemThisNode2 += item
                    item_noCons = [x for x in itemValues.allbounds if x not in itemThisNode2]
                    #print(itemThisNode2)
                if (len(item_noCons) > 0):
                    '''_pParam = copy.deepcopy(sp[vName])
                    _pbounds = [x for x in _pParam.bounds if len(set(item_noCons).intersection(x.bounds)) > 0]
                    for x in _pbounds:
                        x.bounds = list(set(item_noCons).intersection(x.bounds))
                    _pParam.bounds = _pbounds
                    _pParam.allbounds = item_noCons'''
                    _Parent_checked.append([vName, item_noCons, None])
                    lsParentName.append([[vName,item_noCons,None],None])
            del item_noCons, itemThisNode, itemThisNode2, lsOneNode, lsRootNode, lsChildNode, condx
            finaldict = dict()
            lsr = []
            for x in [[x[0],x[1]] for x in _Parent_checked if x[0] not in [x[2] for x in _Parent_checked]]:
                if (x not in lsr):
                    lsr.append(x)
                    xxx = _xxxsingle(x[0], x[0], x[1], None,prefix, lsParentName, sp, con)
                    if (x[0] not in finaldict.keys()):
                        finaldict.update(xxx)
                    else:
                        # print(xxx)
                        finaldict[x[0]].append(xxx[x[0]][0])
            nodes=dict()
            hOpt = _formatsingle(finaldict, None, None, sp, ls,prefix, None, nodes)
            del nodes, finaldict, _Parent_checked, lsParentName
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
import itertools, collections
def ForFullSampling(sp:dict(), con: ConditionalSpace, prefix='value',ifAllSolution=False,
                    random_seed= 0,min_sp=3, n_init_sp=None, max_eval=500,
                    init_sample=10, sample_sp=10, _defratio=0.5, _fair=True):

    np.random.seed(random_seed)
    lsParentName, lsChildEffect, lsFinalSP, lsVarNameinCons, childList = [], [], [], [], []
    _min_sp = min_sp if min_sp != None else int(np.floor(init_sample / sample_sp))
    if ifAllSolution==False:
        _max_sp = int(np.floor(max_eval * _defratio) / sample_sp) if _defratio != None else int(np.floor(init_sample / sample_sp))
    elif isinstance(ifAllSolution,int):
        _max_sp=ifAllSolution
    for i, c in con.conditional.items():
        if (c[1] not in lsVarNameinCons):
            lsVarNameinCons.append(c[1])
        if c[0] not in childList:
            childList.append(c[0])
    lsVarNameinCons =[x for x in lsVarNameinCons if x not in childList]
    lsOneNode = [x for x in sp.keys() if x not in (lsVarNameinCons + childList)]
    _test, _test2 = dict(), dict()
    for vName in lsVarNameinCons + lsOneNode:
        _thisNode = []
        if isinstance(sp[vName], (AlgorithmChoice)):
        #if isinstance(sp[vName],(CategoricalParam,AlgorithmChoice)):
            if len(sp[vName].bounds) > 1:
                _thisNode += [x.bounds for x in sp[vName].bounds]
                _test2[vName] = _thisNode
            else:
                _thisNode += [x for x in sp[vName].bounds[0].bounds]
                _test[vName] = _thisNode
        else:
            _thisNode += [x.bounds for x in sp[vName].bounds]
            _test2[vName] = _thisNode
    _allBounds=dict()
    for i,x in _test2.items():
        _allBounds[i]=[j for i in x for j in i]
    for i,x in _test.items():
        _allBounds[i]=x
    _lsParentName, _i,_a = [], [], dict()
    if ifAllSolution==True:
        _max_sp=9999999
        for i, x in _test2.items():
            _a[i] = [len(x)]
            _i.append(len(x))
        for i, x in _test.items():
            _a[i] = [len(x)]
            _i.append(len(x))
        n_init_sp=np.product(_i)
    else:
        _tmax_sp = n_init_sp if n_init_sp != None else _max_sp
        #_b = {i: [*range(1, len(v) + 1)] for i, v in _test.items()}
        _b = {i: [*range(1, len(v) + 1)] if i in lsVarNameinCons else [1] for i, v in _test.items()}
        _ = [x for x in itertools.product(*list(_b.values())) if
             np.product(x) in [*range(_min_sp, _tmax_sp + 1)]]
        if len(_)>0:
            #_a = {i: [*range(1, len(v) + 1)] for i, v in _test.items()}
            _a={i: [*range(1, len(v) + 1)] if i in lsVarNameinCons else [1] for i, v in _test.items()}
        else:
            _a = {i: [*range(1, len(v) + 1)] for i, v in _test.items()}
        for i, x in _test.items():
            _i.append(len(x))
        for i, x in _test2.items():
            if isinstance(sp[i], (AlgorithmChoice)):
                _a[i] = [*range(1, len(x) + 1)]
                #_a[i] = [len(x)]
            else:
                _a[i]=[1]
            _i.append(len([j for i in x for j in i]))
    #_min_required = np.product(_i) if len(_i) > 0 else 1
    _min_required=1
    if _min_required > _max_sp:
        raise TypeError("Not enought budget for '%s' (at least) sub-search spaces" %
                        _min_required)
    _splitStrategy = []
    #if (n_init_sp != None):
    _splitStrategy = [x for x in itertools.product(*list(_a.values())) if
                      np.product(x) == (n_init_sp if n_init_sp!=None else _max_sp)]
    if (len(_splitStrategy) < 1):
        while len(_splitStrategy)<1 and _min_sp<_max_sp:
            _splitStrategy = [x for x in itertools.product(*list(_a.values())) if
                          np.product(x) ==_max_sp]
            _max_sp=_max_sp-1 if len(_splitStrategy)<1 else _max_sp
    if (len(_splitStrategy) < 1):
        raise TypeError("No spliting solution")
    #_t = sum([np.product(x) for x in _splitStrategy])
    _tarr, _ibest, _ibestValue=[],[],0.00
    for i,x in enumerate(_splitStrategy):
        _pArr=0.00
        for _,_x in enumerate(x):
            _nBounds=_i[_]
            _pro=1-(_x/_nBounds)
            _pArr=_pArr+_pro
        if _ibestValue==_pArr:
            _ibest.append(i)
        elif _ibestValue<_pArr:
            _ibestValue = _pArr
            _ibest=[i]
        else:
            pass
        _tarr.append(_pArr)
    _param_ori = []
    _splitStrategy=[_splitStrategy[i] for i in _ibest]
    if len(_splitStrategy)>1:
        _tarr=[_tarr[i] for i in _ibest]
        _t=np.sum(_tarr)
        _p= [math.floor((x/_t)*100000)/100000 for x in _tarr[:-1]]
        _p.append(1 - sum(_p))
        _choosenstr = _splitStrategy[np.random.choice(len(_splitStrategy),p=_p)]
    else:
        _choosenstr=_splitStrategy[0]
    #p=[np.product(x) / _t for x in _splitStrategy])]
    for i, x in enumerate(_choosenstr):
        _key = list(_a)[i]
        _thisnode = _test[_key] if _key in _test else _test2[_key]
        if x==1:
            _param_ori.append(_key)
            # _lsParentName.append([(_key,[j for i in [x for x in _thisnode] for j in i])])
            _lsParentName.append([(_key, [x for x in _thisnode])])
        elif len(_thisnode) == x:
            _temp=[]
            for _item in _thisnode:
                _temp.append((_key, [_item]))
            _lsParentName.append(_temp)  # [[_key,z] for z in _thisnode]
        else:
            _x=x
            _temp=[]
            _itemthisnode = {i:x for i,x in enumerate(_thisnode)}#[j for i in [x for x in _thisnode] for j in i]
            while len(_itemthisnode) > 0:
                _size = int(np.ceil(len(_itemthisnode) / _x))
                _atemp={i: (1 / (len(x) / len([j for i in list(_itemthisnode.values()) for j in i]))) for i, x in _itemthisnode.items()}
                _asum=sum(_atemp.values())
                _ai=0
                for _ia,_xa in _atemp.items():
                    _ai=_ai+1
                    if _ai >=len(_itemthisnode):
                        _atemp[_ia] = 1-(sum(_atemp.values())-_atemp[_ia])
                    else:
                        _atemp[_ia]=_xa/_asum
                _group = list(np.random.choice(list(_itemthisnode.keys()), _size, replace=False, p=list(_atemp.values()))) if len(
                    _itemthisnode) > _size else list(_itemthisnode.keys())
                #_group =set(list(_thisnode)[:_size]) if len(_thisnode) > _size else _thisnode
                #_thigroup=[j for i in [x for idx,x in _itemthisnode.values() if idx in _group] for j in i]
                #_thisgroup=[j for i in [x for idx,x in _itemthisnode.items() if idx in _group] for j in i]
                _thisgroup=[x for idx, x in _itemthisnode.items() if idx in _group]
                _itemthisnode={i:x for i,x in _itemthisnode.items() if i not in _group}
                #_thisnode = list(set(_thisnode) - set(_thisgroup))
                _temp.append((_key,_thisgroup))
                _x=_x-1
            _lsParentName.append(_temp)
    _combinations=list(itertools.product(*_lsParentName))
    _listSP=[]
    _totalChoices=len([j for i in [x for x in _allBounds.values()] for j in i])
    _lstRatio=[]
    _hpcounted=[]
    _newRange = dict()
    _temp_feeded=dict()
    _isUnEqualBudget = False if init_sample == (1 if sample_sp is None else sample_sp) * len(_combinations) else True
    #_fair = False if _isUnEqualBudget == False else _fair
    for x in _lsParentName:
        #print('===')
        for i, v in x:
            _v = [j for i in v for j in i if isinstance(i, list)]
            _v = v if len(_v) == 0 else _v
            _n={x:i for x,i in zip(_a,_choosenstr)}[i]
            _n=int(len(_combinations)/_n)
            _rangeid=i + ":" + "-".join(str(i) for i in _v)
            _temp_feeded[_rangeid]=0
            _newRange[_rangeid] = _partition(_v, _n, random_seed)
    for x in _combinations:
        _Choice_thisCombi=len([j for i in [z[1] for z in x] for j in i])
        _lstRatio.append(_Choice_thisCombi/_totalChoices)
        _sp=copy.deepcopy(sp)
        _thishp=0
        for i,v in x:
            _v=[j for i in v for j in i if isinstance(i,list)]
            _v=v if len(_v)==0 else _v
            _rangeid = i + ":" + "-".join(str(i) for i in _v)
            _groupIndex=copy.deepcopy(_temp_feeded[_rangeid])
            _temp_feeded[_rangeid]=_groupIndex+1
            _isUserset = True if i in _test2.keys()  and _fair==False  else False
            #if ifAllSolution==False:
            _priGroup = _newRange[_rangeid][_groupIndex]
            _v30 = [x for x in _v if x not in _priGroup]
            _countedByName30, _countedByName70 = dict(), dict()
            h30, h70, h30_min = 0, 0, None
            for _alg in _v30:
                _algLst = _alg if isinstance(_alg, list) else [_alg]
                _thisAlg = len([x for _, x in con.conditional.items() if x[1] == i
                                and len(set(x[2]).intersection(_algLst)) > 0])
                _countedByName30[_alg] = len(_algLst) + _thisAlg
                h30_min = _thisAlg if h30_min == None else min(h30_min, _thisAlg)
                h30 += _thisAlg
            for _alg in _priGroup:
                _algLst = _alg if isinstance(_alg, list) else [_alg]
                _thisAlg = len([x for _, x in con.conditional.items() if x[1] == i
                                and len(set(x[2]).intersection(_algLst)) > 0])
                _countedByName70[_alg] = len(_algLst) + _thisAlg
                h70 += _thisAlg
            z30 = len(_v30)
            z70 = len(_priGroup)
            if _fair==True:

                if  _isUnEqualBudget == True:
                    _alphax = np.mean(list(_countedByName30.values())) if z30 > 0 else 1
                    if z70 > len(_v)/_n:
                        _alpha = 0.35 if _alphax>  np.mean(list(_countedByName70.values()))  else 0.4
                        #_isAnh=True
                        #print('==>',_priGroup, z30, np.mean(list(_countedByName70.values())), _alpha, _alphax)
                    else:
                        _alpha = 0.45 if z30 > 1 and _alphax>  np.mean(list(_countedByName70.values())) else 0.3
                        #print(_priGroup,_alpha,_alphax)
                    _alpha = 1 if _alpha > 1 else _alpha
                    _thishp += (z70 + h70 +z30+ int((h30) * (_alpha)))
                    #_thishp += len(_priGroup) + len(
                    #    [x for _, x in con.conditional.items() if x[1] == i and len(set(x[2]).intersection(_priGroup)) > 0])

                    _p_dict30={i: math.floor((_alpha*(v / (z30+h30)))* 10000) / 10000  for i, v in _countedByName30.items()}
                    _p_remain = (1 - sum(_p_dict30.values())) if z30 > 0 else 1
                    _p_dict70 = {i: math.floor((_p_remain * (v / (z70 + h70))) * 10000) / 10000 for i, v in
                                 _countedByName70.items()}
                    #_p_item_30 = math.floor((0.3 / z30) * 100000) / 100000 if z30 >0 else 0
                    #_p_remain=(1-(_p_item_30*z30)) if z30>0 else 1
                    #_p_item_70 = math.floor(( _p_remain/ z70) * 100000) / 100000

                else:
                    _thishp += (z70 + h70)
                    _p_dict30 = {i: float(0) for i, v in _countedByName30.items()}
                    _p_dict70 = {i: math.floor((v / (z70 + h70)) * 10000) / 10000 for i, v in
                                 _countedByName70.items()}
            else:
                _alpha = 0.5 if z30>z70 else 0.3
                _thishp += int(z70+z30*_alpha)
                _p_dict30 = {i: math.floor((_alpha/z30)* 10000) / 10000 for i, v in _countedByName30.items()}
                _p_remain = (1 - sum(_p_dict30.values())) if z30 > 0 else 1
                _p_dict70 = {i: math.floor((_p_remain/z70) * 10000) / 10000 for i, v in
                             _countedByName70.items()}
                if _isUserset:
                    z = len([j for i in v for j in i]) if _isUserset else len(v)
                    _p_item_70 = math.floor((1 / z) * 10000) / 10000
                '''_thishp+=len(_v)+len([x for _,x in con.conditional.items() if x[1]==i and len(set(x[2]).intersection(_v))>0])
                z = len([j for i in v for j in i]) if _isUserset else len(v)
                _p_item_30 = math.floor((1 / z) * 10000) / 10000
                _p_item_70 = math.floor((1 / z) * 10000) / 10000'''
            _p_remain = 1 - (sum(_p_dict30.values()) + sum(_p_dict70.values()))
            if _p_remain > 0:
                _lowest70 = min(_countedByName70, key=_countedByName70.get)
                #_highest70 = max(_countedByName70, key=_countedByName70.get)
                _p_dict70[_lowest70] = _p_dict70[_lowest70] + _p_remain
            if i in _param_ori:
                continue
            _item=_sp[i]
            _newbounds = []
            for bound in _item.bounds:
                if isinstance(bound, p_paramrange):
                    for _val in bound.bounds:
                        x_bound=copy.deepcopy(bound)
                        x_bound.p = _p_dict30[_val] if _val in _v30 else _p_dict70[_val] if _val in _priGroup else 0
                        _newbounds.append(x_bound)
                else:
                    for _val in bound.bounds:
                        _p = _p_dict30[_val] if _val in _v30 else _p_dict70[_val] if _val in _priGroup else 0
                        _newbounds.append(p_paramrange(bounds=_val, p=_p, default=None, hType="A"))

            _item.bounds=_newbounds
        _hpcounted.append(_thishp)
        _hp_sp=OrginalToHyperopt(_sp,con,prefix)
        _listSP.append(_hp_sp)
    #_returnRatio=_lstRatio / np.mean(_lstRatio)
    #if _fair==True:
        #_hpcounted=np.array(_hpcounted)- (min(_hpcounted)-1)
        #_returnRatio = _hpcounted / sum(_hpcounted)
    _returnRatio=_hpcounted/np.mean(_hpcounted)
    #print(_hpcounted)
    #print(_returnRatio)
    return _listSP, _returnRatio
    #for x in
def OrginalToHyperopt(sp: dict(), con: ConditionalSpace, prefix='value', _fair=True, _todict=False):
    #fullsampling(search_space, con, prefix)
    if con != None:
        condx = [x for _, x in con.AllConditional.items()]
    else:
        condx = []
    if (len(condx) < 1):
        hOpt=dict()
        #ls=search_space._hyperparameters
        for varname,x in sp.items():
            hOpt[x.var_name] = _toHyperoptSyntax(x, x.var_name, x.var_name, prefix, None)
        jointsp=hOpt
    else:
        lsParentName, childList, lsFinalSP, ActiveLst, noCheckForb = [], [], [], [], []
        lsParents = []
        for i, c in con.AllConditional.items():
            _pName, _pValues, _cName = c[1], c[2], c[0]
            if len(set(_pValues).intersection(sp[_pName].allbounds)) < 1:
                continue
            _temp_lsParentName=[]
            for x in sp[_pName].bounds:
                _temp_pValues=[x for x in x.bounds if x in set(_pValues)]#list(set(_pValues).intersection(x.bounds))
                if len(_temp_pValues)>0:
                    _isAlgo = True if isinstance(sp[_pName],AlgorithmChoice) else False
                    _sts=True if _fair==True and len(_temp_pValues)>1 and _isAlgo==True else False
                    _temp_pValues=[[x] for x in _temp_pValues] if _sts==True  else _temp_pValues
                    if _sts==True:
                        _temp_lsParentName.extend(_temp_pValues)
                    else:
                        _temp_lsParentName.append(_temp_pValues)
            for _pValues in _temp_lsParentName:
                if ([[_pName, _pValues], _cName] not in lsParentName):
                    _pbounds2 = [x for x in lsParentName if
                                 x[0][0] == _pName and len(set(_pValues).intersection(x[0][1])) > 0]
                    if len(_pbounds2) > 0:
                        _existBounds = list(np.unique([j for i in [x[0][1] for x in _pbounds2] for j in i]))
                        lsParentName = [x for x in lsParentName if x not in _pbounds2]
                        #_Parent_checked = [x for x in _Parent_checked if x not in _pbounds3]
                    for _x in _pbounds2:
                        _oldDiff, _newDiff, _samelst = [], [], []
                        _pParam = None
                        # _pbounds2.remove(x)
                        _olditem, _oldchild = _x[0], _x[1]
                        _oldDiff += list(set(_olditem[1]) - set(_pValues))
                        _copieditem = [x for x in _pbounds2 if x[0][1] == _oldDiff]
                        if len(_oldDiff) > 0 and len(_copieditem) > 0:
                            if [[_pName, _oldDiff], _oldchild] not in lsParentName:
                                _item = copy.deepcopy(_copieditem[0])
                                _item[1] = _oldchild
                                lsParentName.append(_item)
                                #_Parent_checked.append([_pName, _oldDiff, _oldchild])
                        elif len(_oldDiff) > 0 and [[_pName, _oldDiff], _oldchild] not in lsParentName:
                            lsParentName.append([[_pName, _oldDiff], _oldchild])
                            #_Parent_checked.append([_pName, _oldDiff, _oldchild])
                        try:
                            _newDiff += list(set(_pValues) - set(_existBounds))
                        except:
                            pass
                        _copieditem = [x for x in _pbounds2 if x[0][1] == _newDiff]
                        if len(_newDiff) > 0 and len(_copieditem) > 0:
                            _item = copy.deepcopy(_copieditem[0])
                            _item[1] = _cName
                            lsParentName.append(_item)
                            #_Parent_checked.append([_pName, _newDiff, _cName])
                        elif len(_newDiff) > 0 and [[_pName, _newDiff], _cName] not in lsParentName:
                            lsParentName.append([[_pName, _newDiff], _cName])
                        _samelst += [x for x in _olditem[1] if x in set(_pValues)]#list(set(_pValues).intersection(_olditem[1]))
                        _copieditem = [x for x in _pbounds2 if x[0][1] == _samelst]
                        if len(_samelst) > 0 and len(_copieditem) > 0:
                            if [[_pName, _samelst], _cName] not in lsParentName:
                                _item = copy.deepcopy(_copieditem[0])
                                _item[1] = _cName
                                lsParentName.append(_item)
                            if [[_pName, _samelst], _oldchild] not in lsParentName:
                                _item = copy.deepcopy(_copieditem[0])
                                _item[1] = _oldchild
                                lsParentName.append(_item)
                        elif len(_samelst) > 0:
                            _item = [[_pName, _samelst], _cName]
                            if _item not in lsParentName:
                                lsParentName.append([[_pName, _samelst], _cName])
                            _item = [[_pName, _samelst], _oldchild]
                            if _item not in lsParentName:
                                lsParentName.append([[_pName, _samelst], _oldchild])
                                #_Parent_checked.append([_pName, _samelst, _oldchild])
                    if len(_pbounds2) == 0:
                        lsParentName.append([[_pName, _pValues], _cName])
        del _copieditem
        lsRootNode = list(np.unique(np.array([xp[0] for xp, xc in lsParentName])))

        lsChildNode = [x for _, x in lsParentName]
        lsOneNode = [x for x in sp if x not in (lsRootNode + lsChildNode)]
        # add item no childs and re-order
        _newlsParentName = []
        if (len(lsOneNode) > 0):
            for x in lsOneNode:
                # print(x)
                itemValues = sp[x].allbounds
                if isinstance(sp[x],AlgorithmChoice) and len(itemValues)>1:
                    for item in itemValues:
                        lsParentName.append([[x, [item]], None])
                else:
                    lsParentName.append([[x, itemValues], None])
        for vName in lsRootNode:
            itemValues = sp[vName].allbounds
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
                _isAlgo = True if isinstance(sp[vName], AlgorithmChoice) else False
                _sts = True if _fair == True and len(item_noCons) > 1 and _isAlgo == True else False
                if _sts==True:
                    for item in item_noCons:
                        lsParentName.append([[vName, [item]], None])
                else:
                    lsParentName.append([[vName, item_noCons], None])
        _temp_lsParentName = []
        if _fair == True:
            for xName in lsRootNode:
                _thisNode=sp[xName]
                _isRoot=True if isinstance(_thisNode,AlgorithmChoice) and xName not in lsChildNode else False
                if _isRoot == True:
                    _allbounds = _thisNode.allbounds
                    for _xValue in _allbounds:
                        _temp_lsParentName.extend(
                            [x for x in lsParentName if x[0][0] == xName and set(x[0][1]) == set([_xValue])])
            _temp_lsParentName.extend([x for x in lsParentName if x not in _temp_lsParentName])
            lsParentName = _temp_lsParentName
        del item_noCons, itemThisNode, itemThisNode2, itemValues, childList, condx
        del lsParents, lsOneNode, lsChildNode, lsRootNode
        finaldict = dict()
        lsr = []
        nodes=dict()
        for x in [x for x, _ in lsParentName if x[0] not in [x for _, x in lsParentName]]:
            if (x not in lsr):
                # print(x)
                lsr.append(x)
                xxx = _xxx(x[0], x[0], x[1], None, lsParentName, sp, prefix)
                if (x[0] not in finaldict.keys()):
                    finaldict.update(xxx)
                else:
                    # print(xxx)
                    finaldict[x[0]].append(xxx[x[0]][0])
        if _todict==True:
            return finaldict
        finasp=_format(finaldict,None,None,sp,prefix, None, nodes)
        if(len(finasp)>1):
            jointsp = hp.choice('rootxxxx', [finasp])
        else:
            jointsp=finasp
    del nodes, sp
    del finasp, lsParentName, finaldict
    return jointsp
def __getnewbound(targetvalue, orgHyper:HyperParameter):
    temp=copy.deepcopy(orgHyper)
    if isinstance(temp, (AlgorithmChoice, CategoricalParam)):
        frange = []
        for bound in temp.bounds:
            temp_range = HyperParameter
            if (len(set(targetvalue).intersection(bound.bounds)) > 0):
                temp_range = bound
                _this_values=[x for x in bound.bounds if x in set(targetvalue)]#list(set(targetvalue).intersection(bound.bounds))
                if isinstance(bound, p_paramrange):
                    temp_range.p = round(len(_this_values) * (bound.p / len(bound.bounds)), 5)
                if (isinstance(_this_values, (tuple, list))):
                    temp_range.bounds = [b for b in _this_values]
                else:
                    temp_range.bounds = [_this_values]
                temp_range.default = targetvalue[0] if temp_range.default not in targetvalue else temp_range.default
                frange.append(temp_range)

        temp.bounds = frange
        temp.default = targetvalue[0] if temp.default not in targetvalue else temp.default
        temp.allbounds = [j for i in [x.bounds for x in frange] for j in i]
    return temp
def _newHyper(targetvalue, orgHyper: HyperParameter):
    _thisvalue = copy.deepcopy(orgHyper)
    _pbounds = [x for x in _thisvalue.bounds if len(set(targetvalue).intersection(x.bounds)) > 0]
    for x in _pbounds:
        x.bounds =[x for x in x.bounds if x in set(targetvalue)] #list(set(targetvalue).intersection(x.bounds))
        x.default = targetvalue[0] if x.default not in targetvalue else x.default
    _thisvalue.default = targetvalue[0] if _thisvalue.default not in targetvalue else _thisvalue.default
    _thisvalue.bounds = _pbounds
    _thisvalue.allbounds = targetvalue
    return _thisvalue
def _xxx(rootname, node, value, parent, lsParentName,sp,prefix):
    child_hpa, child_node = _getchilds_single(node, value, lsParentName,sp)
    thisnode=dict()
    isExist=False
    if(len(child_hpa)>0):
        fnode=node
        nodevalue=__getnewbound(value,sp[node])

        thisnode[node]=[{prefix:nodevalue}]
        for child in child_node:
            hpa, _ = _getchilds_single(child[0],child[1], lsParentName,sp)
            childnode=dict()
            if(len(hpa)>0):
                childnode=_xxx(rootname, child[0],child[1],node, lsParentName,sp,prefix)
            else:
                if(child[0] in thisnode[node][0].keys()):
                    childvalue = __getnewbound(child[1], sp[child[0]])
                    childnode[prefix]=childvalue
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
        nodevalue = __getnewbound(value, sp[node])
        if(len([x for x,_ in lsParentName if x[0]==node])>1):
            thisnode[node]=[{prefix:nodevalue}]
        else:
            thisnode[node]=nodevalue
    return thisnode
def XXXXX_getchilds(node, value,lsParentName,sp):
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
def _format(node,alg,parent, sp, prefix, parentvalue, nodes):
    fnode=node
    thisnode=dict()
    #print(node,alg)
    _isP=False
    _P = 0
    if (isinstance(fnode, list)):
        ##hierarchical architecture style
        childlst=list()
        for x in fnode:
            _P=0

            if isinstance(x[prefix].bounds[0],p_paramrange):
                _isP = True
                _P=np.sum([x.p for x in x[prefix].bounds])
            _parentvalue = "".join(str(i) for i in x[prefix].allbounds)

            child=_format(x,alg,alg, sp,prefix,_parentvalue, nodes)
            if _isP:
                childlst.append(tuple([_P,child]))
            else:
                childlst.append(child)
            #childlst
        if _isP:
            thisnode[alg]=hp.pchoice(alg,childlst)
        else:
            thisnode[alg]=hp.choice(alg,childlst)
    elif(isinstance(fnode,dict)):
        childlst= dict()
        for i,v in fnode.items():
            #print('==>',i,v)
            child=_format(v,i,alg, sp,prefix, parentvalue, nodes)
            childlst.update(child)
        return childlst
    else:
        bkalg = alg
        bkparam = alg
        ##update p for child:
        if isinstance(node.bounds[0], p_paramrange):
            _isP = True
            _allP = np.sum([x.p for x in node.bounds])
            #_itemhasP=len([x for x in node.bounds if x.p>0])
            for x in node.bounds:
                x.p = np.floor((x.p/_allP)*10000)/10000 if _allP!=0 else 0
        thisnode[bkalg]=_toHyperoptSyntax(node,alg,parent,prefix, parentvalue, nodes)


    return thisnode
def _toHyperoptSyntax(node:HyperParameter, alg,parent, prefix, parentvalue, nodes):
    #_parentname = (parentvalue if parentvalue!=None else "") +(parent if parent != None else "")
    _parentname="_"+parent if parent != None else ""
    if (alg == prefix):
        bkparam = alg +_parentname+"_"+ "".join(node.allbounds)
        alg #= parent
    else:
        bkparam = alg +_parentname
    if bkparam in nodes:
        return nodes[bkparam]
    var_type = node.var_type
    isPchoice = False
    tmplst = []
    _reValue=None
    if (isinstance(node, (AlgorithmChoice, CategoricalParam))):
        if len(node.bounds) > 1:
            for x in node.bounds:
                temp = None
                _boundname = alg +("_" if _parentname!="" else "_"+_parentname+"_") +"".join(str(i) for i in x.bounds)
                if isinstance(x, p_paramrange):
                    isPchoice = True
                    if (isinstance(x.bounds, (list))):
                        temp = (x.p, hp.choice(_boundname, [a for a in x.bounds]))
                    else:
                        temp = (x.p, x.bounds)
                else:
                    if (isinstance(x.bounds, (list))):
                        temp = (hp.choice(_boundname, [a for a in x.bounds]))
                    else:
                        temp = (x.bounds)
                tmplst.append(temp)
            if isPchoice == True:
                _reValue = hp.pchoice(bkparam, [i for i in tmplst])
            else:
                _reValue = hp.choice(bkparam, [i for i in tmplst])
        else:
            x = node.bounds[0]
            temp = hp.choice(bkparam, x.bounds) if len(x.bounds) > 1 else x.bounds[0]
            _reValue = temp
            '''if isinstance(x, p_paramrange):
                # isPchoice = True
                _reValue = (x.p, {alg:temp})
            else:
                _reValue = {alg:temp}'''
    elif (isinstance(node, FloatParam)):
        floatmap = {"choice": hp.choice, "uniform": hp.uniform, "loguniform": hp.loguniform, "normal": hp.normal,
                    "lognormal": hp.lognormal}
        floatmap_p = {"choice": hp.choice, "uniform": hp.quniform, "loguniform": hp.qloguniform, "normal": hp.qnormal,
                      "lognormal": hp.qlognormal}
        utype = "uniform" if node.scale == None else node.scale
        if len(node.bounds) > 1:
            for x in node.bounds:
                temp = None
                _boundname = alg +("_" if _parentname!="" else "_"+_parentname+"_" )+ "".join(str(i) for i in x.bounds)
                try:
                    if x.type == "list":
                        utype = "choice"
                except:
                    pass
                if isinstance(x, p_paramrange):
                    isPchoice = True
                    if utype == "list":
                        temp = (x.p, hp.choice(_boundname, x.bounds))
                    else:
                        if (x.q != None):
                            temp = (x.p, floatmap_p[utype](_boundname, x.lower, x.upper, x.q))
                        else:
                            temp = (x.p, floatmap[utype](_boundname, x.lower, x.upper))
                else:
                    temp = (floatmap[utype](_boundname, x.lower, x.upper))
                tmplst.append(temp)
            if isPchoice == True:
                _reValue = hp.pchoice(bkparam, [i for i in tmplst])
            else:
                _reValue = hp.choice(bkparam, [i for i in tmplst])
        else:
            x = node.bounds[0]
            if x.type == "list":
                _reValue = hp.choice(bkparam, x.bounds)
            else:
                _reValue = floatmap[utype](bkparam, x.lower, x.upper)
    elif (isinstance(node, IntegerParam)):
        if len(node.bounds) > 1:
            for x in node.bounds:
                temp = None
                _boundname = alg +("_" if _parentname!="" else "_"+_parentname+"_")+ "".join(str(i) for i in x.bounds)
                if isinstance(x, p_paramrange):
                    isPchoice = True
                    temp = (x.p, hp.randint(_boundname, x.lower, x.upper)) if x.type == "range" \
                        else (x.p, hp.choice(_boundname, x.bounds))
                else:
                    temp = (hp.randint(_boundname, x.lower, x.upper)) if x.type == "range" \
                        else (hp.choice(_boundname, x.bounds))
                tmplst.append(temp)
            if isPchoice == True:
                _reValue= hp.pchoice(bkparam, [i for i in tmplst])
            else:
                _reValue = hp.choice(bkparam, [i for i in tmplst])
        else:
            x = node.bounds[0]
            _reValue = hp.randint(bkparam, x.lower, x.upper) if x.type == "range" \
                else hp.choice(bkparam, x.bounds)
    nodes[bkparam]=_reValue
    return _reValue

def _xxxsingle(rootname, node, value, parent, prefix, lsParentName,sp, con):
    child_hpa, child_node = _getchilds_single(node, value, lsParentName,sp)
    thisnode=dict()
    isExist=False
    #_thisvalue=value
    #_thisvalue = [x for x, _ in lsParentName if x[0] == node and x[1] == value][0][1]
    '''_thisvalue = copy.deepcopy(sp[node])
    _pbounds = [x for x in _thisvalue.bounds if len(set(value).intersection(x.bounds)) > 0]
    for x in _pbounds:
        x.bounds = list(set(value).intersection(x.bounds))
        x.default = value[0] if x.default not in value else x.default
    _thisvalue.default= value[0] if _thisvalue.default not in value else _thisvalue.default
    _thisvalue.bounds = _pbounds
    _thisvalue.allbounds = value'''
    _thisvalue=__getnewbound(value,sp[node])
    if(len(child_hpa)>0):
        fnode=node
        thisnode[node]=[{prefix:_thisvalue}]
        for child in child_node:
            hpa, _ = _getchilds_single(child[0],child[1], lsParentName,sp)
            childnode=dict()
            if(len(hpa)>0):
                childnode=_xxxsingle(rootname, child[0],child[1],node,prefix,
                                     lsParentName,sp, con)
            else:
                _index = prefix if child[0] in thisnode[node][0].keys() else child[0]
                _newchildnode=child[-1] if isinstance(child[-1],HyperParameter) else _newHyper(child[1], sp[child[0]])

                childnode[_index] = _newchildnode

            if (child[0] in thisnode[node][0].keys()):
                if(list(childnode.keys())[0]!=prefix):
                    first_value = next(iter(childnode.values()))
                    thisnode[node][0][child[0]].append(first_value[0])
                else:
                    thisnode[node][0][child[0]].append(childnode)
            else:
                thisnode[node][0].update(childnode)
    else:
        #_thisvalue = [x for x, _ in lsParentName if x[0] == node and x[1] == value][0]
        if(len([x for x,_ in lsParentName if x[0]==node])>1):
            thisnode[node]=[{prefix:_thisvalue}]
        else:
            if(len([x[2] for _,x in con.conditional.items() if x[1]==node])>0):
                thisnode[node] = [{prefix: _thisvalue}]
            else:
                thisnode[node]=_thisvalue
    return thisnode
def _getchilds_single(node, value,lsParentName,sp):
    child_node=[]
    child_hpa=[_ for x,_ in lsParentName if x[0]==node and x[1]==value and _!=None]
    for child in child_hpa:
        #print(child,[x for x,_ in lsParentName if x[0]==child])
        if (len([x for x,_ in lsParentName if x[0]==child])>0):
            thisChild=[]
            for x in [x for x,_ in lsParentName if x[0] == child]:
                if x not in thisChild:
                    thisChild.append(x)
            child_node.extend(thisChild)
        else:
            #print(child)
            item=sp[child]
            thisChild=[child,item]
        #print(thisChild)
            child_node.append(thisChild)
    return child_hpa, child_node
def _formatsingle(node,alg,parent, sp,productspace, prefix='value',parentvalue=None, nodes=None):
    fnode=node
    thisnode=dict()
    #print(node,alg)
    _isP = False
    _P = 0
    if (isinstance(fnode, list)):
        childlst=[]
        for x in fnode:
            _P = 0
            if isinstance(x[prefix].bounds[0],p_paramrange):
                _isP = True
                _P=x[prefix].bounds[0].p
            _parentvalue="".join(str(i) for i in x[prefix].allbounds)
            child=_formatsingle(x,alg,alg, sp,productspace,prefix, _parentvalue, nodes)
            if _isP:
                childlst.append(tuple([_P,child]))
            else:
                childlst.append(child)
            #childlst.append(child)
        #thisnode[alg]=hp.choice(alg,childlst)
        if _isP:
            thisnode[alg]=hp.pchoice(alg,childlst)
        else:
            thisnode[alg]=hp.choice(alg,childlst)
    elif(isinstance(fnode,dict)):
        childlst=dict()
        for i,v in fnode.items():
            #print('==>',i,v)
            #_parentvalue = "".join(str(i) for i in v[prefix].allbounds)
            child=_formatsingle(v,i,alg, sp,productspace,prefix, parentvalue, nodes)
            childlst.update(child)
        return childlst
    else:
        bkalg = alg

        thisnode[bkalg] = _toHyperoptSyntax(node, alg, parent, prefix,parentvalue, nodes)
    return thisnode
def _partition (list_org, n, seed):
    list_in=copy.deepcopy(list_org)
    np.random.seed(seed)
    _s=len(list_in)
    _results=[]
    if _s>=n:
        np.random.shuffle(list_in)
        _results= [list_in[i::n] for i in range(n)]
    elif _s<n and _s>1:
        np.random.shuffle(list_in)
        _res=[list_in[i::_s] for i in range(_s)]
        _ran=list(np.random.choice(list_in,size=n-_s))
        _res.extend([[x] for x in _ran])
        _results= _res
    elif _s==1:
        _results= [list_in]*n
    else:
        _results= [list_in]
    np.random.shuffle(_results)
    return _results

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