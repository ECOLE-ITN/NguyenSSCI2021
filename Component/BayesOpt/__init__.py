from __future__ import absolute_import
from bayes_optim import Surrogate
#from BayesOpt.BayesOpt import BO
from bayes_optim import BO
import bayes_optim.AcquisitionFunction as InfillCriteria
from bayes_optim.Surrogate import RandomForest
from bayes_optim.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace
from bayes_optim import BayesOpt
from bayes_optim.base import Solution
import BanditOpt.ParamExtension as ext
SearchSpace.__init__=ext.init_SearchSpace
NominalSpace.__init__=ext.init_NominalSpace
ContinuousSpace.__init__=ext.init_ContinuousSpace
OrdinalSpace.__init__=ext.init_OrdinalSpace
BO.tell=ext.BayesOpt_tell
__all__ = ['BO','BayesOpt','RandomForest', 'InfillCriteria', 'Surrogate', 'ContinuousSpace', 'NominalSpace', 'OrdinalSpace', 'SearchSpace','Solution']
