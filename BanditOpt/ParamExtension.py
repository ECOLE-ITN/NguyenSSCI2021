from Component.BayesOpt import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace
import numpy as np
import copy
from BayesOpt.BayesOpt import BO
from BayesOpt.base import Solution


def rebuild(hyperparameter):
    if (isinstance(hyperparameter, OrdinalSpace)):
        pass
    elif (isinstance(hyperparameter, NominalSpace)):
        #hyperparameter.levels = [np.array(b) for b in hyperparameter.bounds]
        #hyperparameter._levels = [np.array(b) for b in hyperparameter.bounds]
        hyperparameter.levels = {i: hyperparameter.bounds[i] for i in hyperparameter.id_N}
        hyperparameter._levels = {i: hyperparameter.bounds[i] for i in hyperparameter.id_N}
        hyperparameter._n_levels = {i: len(hyperparameter.bounds[i]) for i in hyperparameter.id_N}
        #hyperparameter._n_levels = [len(l) for l in hyperparameter._levels]
    elif (isinstance(hyperparameter, ContinuousSpace)):
        pass
    return hyperparameter


def check_configuration(self, X):
    """
            check for the duplicated solutions, as it is not allowed
            for noiseless objective functions
            2020/7/23: check Forbidden
            """
    if not isinstance(X, Solution):
        X = Solution(X, var_name=self.var_names)
    N = X.N
    if hasattr(self, 'data'):
        X = X + self.data
    _ = []
    for i in range(N):
        x = X[i]
        x_dict= dict(zip(self.var_names, x))
        isFOB=False
        for fname,fvalue in self._forbidden.forbList.items():
            hp_left = [(key,value) for (key, value) in x_dict.items() if
                       key == fvalue.left and len(set([value]).intersection(fvalue.leftvalue)) > 0]
            hp_right = [(key,value) for (key, value) in x_dict.items() if
                       key == fvalue.right and len(set([value]).intersection(fvalue.rightvalue)) > 0]
            if(len(hp_left)>0 and len(hp_right)>0):
                isFOB=True
        idx = np.arange(len(X)) != i
        CON = np.all(np.isclose(np.asarray(X[idx][:, self.r_index], dtype='float'),
                                np.asarray(x[self.r_index], dtype='float')), axis=1)
        INT = np.all(X[idx][:, self.i_index] == x[self.i_index], axis=1)
        CAT = np.all(X[idx][:, self.d_index] == x[self.d_index], axis=1)
        if not any(CON & INT & CAT):
            if (isFOB == False):
                _ += [i]
    return X[_]


def BOrun_todict(self):
    while not self.check_stop():
        self.step()

    return self.xopt.to_dict(), self.xopt.fitness, self.stop_dict


def _set_levels(self):
    """Set categorical levels for all nominal variables
    """
    if hasattr(self, 'id_N') and len(self.id_N) > 0:
        self.levels = {i: self.bounds[i] for i in self.id_N}
        self._n_levels = {i: len(self.bounds[i]) for i in self.id_N}
    else:
        self.levels, self._n_levels = None, None
