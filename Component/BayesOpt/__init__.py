from BayesOpt import Surrogate
#from BayesOpt.BayesOpt import BO
from BayesOpt.InfillCriteria import InfillCriteria
from BayesOpt.Surrogate import RandomForest
from BayesOpt.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace
import BanditOpt.ParamExtension as ext
SearchSpace.__init__=ext.init_SearchSpace
NominalSpace.__init__=ext.init_NominalSpace
ContinuousSpace.__init__=ext.init_ContinuousSpace
OrdinalSpace.__init__=ext.init_OrdinalSpace
from .bayesopt import BayesOpt
__all__ = ['BayesOpt','RandomForest', 'InfillCriteria', 'Surrogate', 'ContinuousSpace', 'NominalSpace', 'OrdinalSpace', 'SearchSpace']
