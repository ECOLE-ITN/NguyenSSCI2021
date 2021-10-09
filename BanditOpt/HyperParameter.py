import numpy as np
from numpy.random import randint, rand
from abc import abstractmethod
from BanditOpt.ParamRange import paramrange,p_paramrange,one_paramrange

class HyperParameter(object):
    def __init__(self, bounds, var_name, name,cutting=None, default=None, hType="C"):
        if isinstance(bounds,(list,tuple,)):
            _thisbound = list()
            _joinbound = list()
            _hasP_param=False
            _allbounds=[]
            _thisDef = None
            if(any(isinstance(x,(tuple,list)) for x in bounds)==False):
                _thisbound.append(paramrange(bounds,default=default,hType=hType))
                if(hType=="I"):
                    if(len(bounds)==2):
                        _allbounds=[*range(bounds[0],bounds[1])]
                    else:
                        _allbounds=bounds
            else:
                #if(any(isinstance(x,tuple))for x in bounds):
                _sumP=sum([x[1] for x in bounds if isinstance(x, tuple)])
                _hasP_param=True if _sumP>0 else False
                _itemNoP=len([x for x in bounds if isinstance(x,tuple)==False])
                for bound in bounds:
                    if (hType == "I"):
                        if isinstance(bound,tuple):
                            _p = bound[1]
                            if (isinstance(bound[0], list) and len(bound[0]) == 2):
                                _lower = bound[0][0]
                                _upper = bound[0][1]
                                _intbound = [_lower, _upper]
                                _allbounds.extend([*range(_lower, _upper)])
                                bound = (_intbound, _p)
                                if default in _intbound:
                                    _thisDef = default
                            else:
                                if isinstance(bound[0],list)==False:
                                    bound=([bound[0]],_p)
                                _allbounds.extend(bound[0])
                                if default in bound[0]:
                                    _thisDef = default
                            _thisbound.append(p_paramrange(*bound, default=_thisDef, hType=hType))
                        else:
                            if (isinstance(bound,list) and len(bound) == 2):
                                _lower = bound[0]
                                _upper = bound[1]
                                _allbounds.extend([*range(_lower, _upper)])
                                _intbound = [_lower, _upper]
                                if default in _intbound:
                                    _thisDef=default
                                bound=_intbound
                            else:
                                if (isinstance(bound,list)==False):
                                    bound=[bound]
                                _allbounds.extend(bound)
                                    #_joinbound.append(bound)
                                if default in bound:
                                    _thisDef = default
                            if _hasP_param:
                                _p=round(((1-_sumP)/_itemNoP),5)
                                _thisbound.append(p_paramrange(bounds=bound,p=_p, default=_thisDef, hType=hType))
                            else:
                                _thisbound.append(paramrange(bounds=bound, default=_thisDef, hType=hType))
                    else:
                        if isinstance(bound,tuple):
                            _thisbound.append(p_paramrange(*bound,hType=hType))
                        else:
                            if (isinstance(bound, list) == False):
                                bound = [bound]
                            _allbounds.extend(bound)
                            _thisDef = default if default in bound else bound[0]
                            _p = round(((1 - _sumP) / _itemNoP), 5) if _hasP_param else None
                            _thisItem = p_paramrange(bounds=bound, p=_p, default=_thisDef,
                                                     hType=hType) if _hasP_param else \
                                paramrange(bounds=bound, default=_thisDef, hType=hType)
                            _thisbound.append(_thisItem)
                            '''
                            if hType=="A":                                
                                _p = round(_p / len(bound), 5)
                                for x in bound:
                                    _thisDef=default if default ==x else None
                                    _thisbound.append(p_paramrange(bounds=[x], p=_p, default=_thisDef, hType=hType))                                    
                            else:
                                if _hasP_param:
                                    #_p = round(((1 - _sumP) / _itemNoP), 5)
                                    _thisbound.append(p_paramrange(bounds=bound, p=_p, default=_thisDef, hType=hType))
                                else:
                                    _thisbound.append(paramrange(bounds=bound, default=_thisDef, hType=hType))
                                    '''
                            '''if(isinstance(bound,list)):
                                _joinbound.extend(bound)
                            else:
                                _joinbound.append(bound)
                    #elif isinstance(bound,list):
                    #   pbounds.append(paramrange(bound))
                    #else:
                    #    pbounds.append(paramrange(*bound))
                if(len(_joinbound)>0):
                    thisDef=None
                    if(default in _joinbound):
                        thisDef=default
                    if(iP>0.0):
                        thisP=round(1-iP,5)
                        if(thisP<0):
                            thisP=0.1
                        _thisbound.append(p_paramrange(bounds=_joinbound,p=thisP,scalerule=None, default=thisDef,hType=hType))
                    else:
                        _thisbound.append(paramrange(_joinbound,thisDef,hType=hType))'''
            if(hType in ["C","A"]):
                self.allbounds=[j for i in [x.bounds for x in _thisbound] for j in i]
            elif hType =="I":
                self.allbounds=_allbounds
            else:
                self.allbounds = [x.bounds for x in _thisbound]
            self.bounds = _thisbound
        else:
            self.allbounds = [bounds]
            self.bounds = [paramrange(bounds,hType)]

            if (default in self.bounds[0].bounds and self.bounds[0].default==None):
                self.bounds[0].default=default
            pass
        '''else:
            if hasattr(bounds[0], '__iter__') and not isinstance(bounds[0], str):
                self.bounds = [tuple(b) for b in bounds]
            else:
                self.bounds = [tuple(bounds)]
                '''

        self.name = name
        self.var_type = hType
        self.default= default
        self.cutting = cutting
        '''dim=len(self.bounds)
        if var_name is not None:
            if isinstance(var_name, str):
                if dim > 1:
                    var_name = [var_name + '_' + str(_) for _ in range(dim)]
                else:
                    var_name = [var_name]
            assert len(var_name) == dim'''
        self.var_name = var_name
    def _set_index(self):
        self.C_mask = np.asarray(self.var_type) == 'C'  # Continuous
        self.O_mask = np.asarray(self.var_type) == 'O'  # Ordinal
        self.N_mask = np.asarray(self.var_type) == 'N'  # Nominal

        self.id_C = np.nonzero(self.C_mask)[0]
        self.id_O = np.nonzero(self.O_mask)[0]
        self.id_N = np.nonzero(self.N_mask)[0]
class FloatParam(HyperParameter):
    """Continuous (real-valued) hyperparameter
        """
    def __init__(
        self,
        bounds,
        var_name='f',
        name=None,
        cutting=None,
        default=None,
        scale = None
        ):
        #bounds
        super(FloatParam, self).__init__(bounds, var_name, name, cutting, default, hType="F")
        self.scale=scale
class IntegerParam(HyperParameter):
    """Ordinal (integer) hyperparameter
    """
    def __init__(self,
        bounds,
        var_name='i',
        name = None,
        cutting=None,
        default=None):
        super(IntegerParam,self).__init__(bounds, var_name, name, cutting, default, hType="I")
class AlgorithmChoice(HyperParameter):
    def __init__(self, bounds, var_name='c', name=None, cutting=None,
        default=None):
        #bounds=self._get_unique_bounds(bounds)
        super(AlgorithmChoice,self).__init__(bounds,var_name,name,cutting, default, hType="A")

class CategoricalParam(HyperParameter):
    """Nominal (discrete) Search Space
        """
    def __init__(self, bounds, var_name='c', name=None, cutting=None,
        default=None):
        #bounds=self._get_unique_bounds(bounds)
        super(CategoricalParam,self).__init__(bounds,var_name,name,cutting, default, hType="C")

if __name__ == '__main__':
    #import HyperParameter,FloatParam, CategoricalParam, IntegerParam
    from BanditOpt.ConditionalSpace import ConditionalSpace
    from BanditOpt.ConfigSpace import ConfigSpace
    cs = ConfigSpace()
    alg_namestr = CategoricalParam([("SVM",0.4), "RF",['LR','DT']], "alg_namestr")
    test = CategoricalParam(("A","B"), "test", default="A")
    C = FloatParam([1e-2, 100], "C")
    degree = IntegerParam([([1, 2],0.1),([3,5],.44),[6,10],12], 'degree')
    f=FloatParam([(0.01,0.5),[0.02,100]],"testf")
    abc=CategoricalParam(range(1,50,2),"abc")
    test.bounds.append(paramrange([1,5],hType='F'))
    test.allbounds.append([1,5])
    print(abc)


