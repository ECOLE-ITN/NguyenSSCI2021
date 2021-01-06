from Component.BayesOpt import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace
import numpy as np
from copy import copy, deepcopy
#from BayesOpt.BayesOpt import BO
#from BayesOpt.base import Solution
from Component.BayesOpt import Solution


###Search space
def init_SearchSpace(self, bounds, var_name, name,  random_seed=None, default=None):
    """Search Space Base Class

    Parameters
    ---------
    bounds : (list of) list,
        lower and upper bound for continuous/ordinal parameter type
        categorical values for nominal parameter type.
        The dimension of the space is determined by the length of the
        nested list
    var_name : (list of) str,
        variable name per dimension. If only a string is given for multiple
        dimensions, variable names are created by appending counting numbers
        to the input string.
    name : str,
        search space name. It is typically used as the grouping variable
        when converting the Solution object to dictionary, allowing for
        vector-valued search parameters. See 'to_dict' method below.
    default: default value,
        will use in impute strategy
    Attributes
    ----------
    dim : int,
        dimensinality of the search space
    bounds : a list of lists,
        each sub-list stores the lower and upper bound for continuous/ordinal variable
        and categorical values for nominal variable
    levels : a list of lists,
        each sub-list stores the categorical levels for every nominal variable. It takes
        `None` value when there is no nomimal variable
    precision : a list of double,
        the numerical precison (granularity) of continuous parameters, which usually
        very practical in real-world applications
    var_name : a list of str,
        variable names per dimension
    var_type : a list of str,
        variable type per dimension, 'C': continuous, 'N': nominal, 'O': ordinal
    C_mask : a bool array,
        the mask array for continuous variables
    O_mask : a bool array,
        the mask array for integer variables
    N_mask : a bool array,
        the mask array for discrete variables
    id_C : an int array,
        the index array for continuous variables
    id_O : an int array,
        the index array for integer variables
    id_N : an int array,
        the index array for discrete variables
    """
    if hasattr(bounds[0], '__iter__') and not isinstance(bounds[0], str):
        self.bounds = [tuple(b) for b in bounds]
        if (default == None):
            self.default = bounds[0][0]
        else:
            self.default = default
    else:
        self.bounds = [tuple(bounds)]
        if (default == None):
            self.default = np.mean(bounds)
        else:
            self.default = default
    self.dim = len(self.bounds)
    self.name = name
    self.random_seed = random_seed
    self.var_type = None
    self.levels = None
    #self.precision = None
    self.precision = {}
    self.scale = {}
    if var_name is not None:
        if isinstance(var_name, str):
            if self.dim > 1:
                var_name = [var_name + '_' + str(_) for _ in range(self.dim)]
            else:
                var_name = [var_name]
        assert len(var_name) == self.dim
        self.var_name = var_name


def init_NominalSpace(self, levels, var_name='d', name=None, default=None):
    #update 30/12/2020
    #levels = np.atleast_2d(levels)
    #levels = [np.unique(l).tolist() for l in levels]
    levels = self._get_unique_levels(levels)
    super(NominalSpace, self).__init__(levels, var_name, name, default)
    self.var_type = ['N'] * self.dim
    self._levels = [np.array(b) for b in self.bounds]
    self._set_index()
    self._set_levels()


def init_ContinuousSpace(self, bounds, var_name='r', name=None, precision=None,scale=None, default=None):
    super(ContinuousSpace, self).__init__(bounds, var_name, name, default)
    self.var_type = ['C'] * self.dim
    self._bounds = np.atleast_2d(self.bounds).T
    self._set_index()

    # set up precisions for each dimension
    if hasattr(precision, '__iter__'):
        assert len(precision) == self.dim
        self.precision = {i: precision[i] \
                          for i in range(self.dim) if precision[i] is not None}
    else:
        if precision is not None:
            self.precision = {i: precision for i in range(self.dim)}
    ##update 30/12/2020: BayesOpt change to bayes_optim
        # set up the scale for each dimension
        if scale is not None:
            if isinstance(scale, str):
                scale = [scale] * self.dim
            elif hasattr(scale, '__iter__'):
                assert len(scale) == self.dim

            self.scale = {
                i: scale[i] for i in range(self.dim) if scale[i] is not None
            }

        for i, s in self.scale.items():
            lower, upper = self.bounds[i]
            self.bounds[i] = (TRANS[s](lower), TRANS[s](upper))

        self._bounds = np.atleast_2d(self.bounds).T
    assert all(self._bounds[0, :] < self._bounds[1, :])


def init_OrdinalSpace(self, bounds, var_name='i', name=None, default=None):
    super(OrdinalSpace, self).__init__(bounds, var_name, name, default)
    self.var_type = ['O'] * self.dim
    self._lb, self._ub = zip(*self.bounds)  # for sampling
    assert all(np.array(self._lb) < np.array(self._ub))
    self._set_index()


def rebuild(hyperparameter):
    if (isinstance(hyperparameter, OrdinalSpace)):
        pass
    elif (isinstance(hyperparameter, NominalSpace)):
        # hyperparameter.levels = [np.array(b) for b in hyperparameter.bounds]
        # hyperparameter._levels = [np.array(b) for b in hyperparameter.bounds]
        hyperparameter.levels = {i: hyperparameter.bounds[i] for i in hyperparameter.id_N}
        hyperparameter._levels = {i: hyperparameter.bounds[i] for i in hyperparameter.id_N}
        hyperparameter._n_levels = {i: len(hyperparameter.bounds[i]) for i in hyperparameter.id_N}
        if (hyperparameter.default not in hyperparameter.bounds[0]):
            hyperparameter.default = hyperparameter.bounds[0][0]
        # hyperparameter._n_levels = [len(l) for l in hyperparameter._levels]
    elif (isinstance(hyperparameter, ContinuousSpace)):
        pass
    return hyperparameter


def formatCandidate(self, data):
    if self._conditional==None:
        return data
    lsParentName, childList, lsFinalSP, ActiveLst, noCheckForb, var_names = [], [], [], [], [], []
    var_names=self.var_names
    for i, con in self._conditional.conditional.items():
        if ([con[1], con[2], con[0]] not in lsParentName):
            lsParentName.append([con[1], con[2], con[0]])
        if (con[0] not in childList):
            childList.append(con[0])
    lsRootNode = [x for x in var_names if x not in childList]
    for root in lsRootNode:
        rootvalue = data[root]
        ActiveLst.append(root)
        for node, value in [(x[2], x[1]) for x in lsParentName if x[0] == root and rootvalue in x[1]]:
            value = data[node]
            ActiveLst.append(node)
            nodeChilds = [(x[2], x[1]) for x in lsParentName if x[0] == node and value in x[1]]
            while (len(nodeChilds) > 0):
                childofChild = []
                for idx, child in enumerate(nodeChilds):
                    childvalue = data[child[0]]
                    childofChild.extend([(x[2], x[1]) for x in lsParentName if x[0] == child[0] and childvalue in x[1]])
                    ActiveLst.append(child[0])
                    # del nodeChilds[idx]
                nodeChilds.clear()
                if (len(childofChild) > 0):
                    nodeChilds = childofChild
    newX = data
    if self._eval_type == 'dict':
        newX = dict()
        for x in ActiveLst:
            newX[x] = data[x]
    else:
        pass
    return newX


def imputation(conditional, x, var_names, defaultvalue):
    lsParentName, childList, lsFinalSP, ActiveLst, noCheckForb = [], [], [], [], []
    for i, con in conditional.AllConditional.items():
        if ([con[1], con[2], con[0]] not in lsParentName):
            lsParentName.append([con[1], con[2], con[0]])
        if (con[0] not in childList):
            childList.append(con[0])
    lsRootNode = [x for x in var_names if x not in childList]
    # indextoremove=[]
    for root in lsRootNode:
        rootvalue = x[var_names.index(root)]
        ActiveLst.append(root)
        for node, value in [(x[2], x[1]) for x in lsParentName if x[0] == root and rootvalue in x[1]]:
            value = x[var_names.index(node)]
            ActiveLst.append(node)
            nodeChilds = [(x[2], x[1]) for x in lsParentName if x[0] == node and value in x[1]]
            while (len(nodeChilds) > 0):
                childofChild = []
                for idx, child in enumerate(nodeChilds):
                    childvalue = x[var_names.index(child[0])]
                    childofChild.extend([(x[2], x[1]) for x in lsParentName if x[0] == child[0] and childvalue in x[1]])
                    ActiveLst.append(child[0])
                    #del nodeChilds[idx]
                nodeChilds.clear()
                if (len(childofChild) > 0):
                    nodeChilds = childofChild
    noCheckForb = [x for x in var_names if x not in ActiveLst]
    for node in noCheckForb:
        x[var_names.index(node)] = defaultvalue[node]

    return x, noCheckForb

def evaluate(self, X):
    """Evaluate the candidate points and update evaluation info in the dataframe
    """
    X = [self.formatCandidate(x) for x in X]
    # Parallelization is handled by the objective function itself
    if self.parallel_obj_fun is not None:
        func_vals = self.parallel_obj_fun(X)
    else:
        if self.n_job > 1:  # or by ourselves..
            func_vals = Parallel(n_jobs=self.n_job)(delayed(self.obj_fun)(x) for x in X)
        else:  # or sequential execution
            func_vals = [self.obj_fun(x) for x in X]

    return func_vals
'''def evaluate(self, data):
    #"""Evaluate the candidate points and update evaluation info in the dataframe
    #"""
    N = len(data)
    if self._eval_type == 'list':
        X = [x.tolist() for x in data]
    elif self._eval_type == 'dict':
        X = [self._space.to_dict(x) for x in data]

    # Parallelization is handled by the objective function itself
    X = [self.formatCandidate(x) for x in X]
    if self.parallel_obj_fun is not None:
        func_vals = self.parallel_obj_fun(X)
    else:
        if self.n_job > 1:
            func_vals = Parallel(n_jobs=self.n_job)(delayed(self.obj_fun)(x) for x in X)
        else:
            func_vals = [self.obj_fun(x) for x in X]

    self.eval_count += N
    return func_vals
'''

def check_configuration(self, X):
    """
            check for the duplicated solutions, as it is not allowed
            for noiseless objective functions
            d.a.nguyen: check Forbidden & check conditional
            """
    # X_array=copy.deepcopy(X)
    if not isinstance(X, Solution):
        X = Solution(X, var_name=self.var_names)
    N = X.N
    if hasattr(self, 'data'):
        X = X + self.data
    _ = []
    defaultvalue = {i: x.default for (i, x) in self._hyperparameters._hyperparameters.items()}
    for i in range(N):
        x = X[i]
        idx = np.arange(len(X)) != i
        CON = np.all(np.isclose(np.asarray(X[idx][:, self.r_index], dtype='float'),
                                np.asarray(x[self.r_index], dtype='float')), axis=1)
        INT = np.all(X[idx][:, self.i_index] == x[self.i_index], axis=1)
        CAT = np.all(X[idx][:, self.d_index] == x[self.d_index], axis=1)
        if not any(CON & INT & CAT):
            """d.a.nguyen: add check conditional and forbidden here"""
            isBandit = self._isBandit
            noChecklst = []
            if (self._conditional != None):
                x, noChecklst = imputation(self._conditional, x, self.var_names, defaultvalue)
            else:
                pass
            ##Check Forbidden
            # noCheckid=[i for (i, v) in enumerate(self.var_names) if v in noChecklst]
            x_dict = dict(zip(self.var_names, x))
            x_dict = dict((i, v) for (i, v) in x_dict.items() if i not in noChecklst)
            isFOB = False
            if (self._forbidden != None):
                for fname, fvalue in self._forbidden.forbList.items():
                    hp_left = [(key, value) for (key, value) in x_dict.items() if
                               key == fvalue.left and len(set([value]).intersection(fvalue.leftvalue)) > 0]
                    hp_right = [(key, value) for (key, value) in x_dict.items() if
                                key == fvalue.right and len(set([value]).intersection(fvalue.rightvalue)) > 0]
                    hp_add1, hp_add2 = [], []
                    if (fvalue.ladd1 != None):
                        hp_add1 = [(key, value) for (key, value) in x_dict.items() if
                                   key == fvalue.ladd1 and len(set([value]).intersection(fvalue.ladd1value)) > 0]
                    if (fvalue.ladd2 != None):
                        hp_add2 = [(key, value) for (key, value) in x_dict.items() if
                                   key == fvalue.ladd2 and len(set([value]).intersection(fvalue.ladd2value)) > 0]
                    if (fvalue.ladd1 != None and fvalue.ladd2 != None):
                        if (len(hp_left) > 0 and len(hp_right) > 0 and len(hp_add1) > 0 and len(hp_add2) > 0):
                            isFOB = True
                    elif (fvalue.ladd1 != None):
                        if (len(hp_left) > 0 and len(hp_right) > 0 and len(hp_add1) > 0):
                            isFOB = True
                    else:
                        if (len(hp_left) > 0 and len(hp_right) > 0):
                            isFOB = True

            """d.a.nguyen: update X based on conditional ::: end"""
            if (isFOB == False):
                _ += [i]

    return X[_]

'''
def BOrun_todict(self):
    while not self.check_stop():
        self.step()

    return self.xopt.to_dict(), self.xopt.fitness, self.stop_dict
'''

def _set_levels(self):
    """Set categorical levels for all nominal variables
    """
    if hasattr(self, 'id_N') and len(self.id_N) > 0:
        self.levels = {i: self.bounds[i] for i in self.id_N}
        self._n_levels = {i: len(self.bounds[i]) for i in self.id_N}
    else:
        self.levels, self._n_levels = None, None


def BayesOpt_tell(self, X, func_vals, warm_start=False):
    """Tell the BO about the function values of proposed candidate solutions

    Parameters
    ----------
    X : List of Lists or Solution
        The candidate solutions which are usually proposed by the `self.ask` function
    func_vals : List/np.ndarray of reals
        The corresponding function values
    """
    X = self._to_geno(X)

    if warm_start:
        msg = 'warm-starting from {} points:'.format(len(X))
    elif self.iter_count == 0:
        msg = 'initial DoE of size {}:'.format(len(X))
    else:
        msg = 'iteration {}, {} infill points:'.format(self.iter_count, len(X))

    self._logger.info(msg)
    #         X_ = self._to_pheno(X)

    for i in range(len(X)):
        X[i].fitness = func_vals[i]
        X[i].n_eval += 1

        if not warm_start:
            self.eval_count += 1

        self._logger.info(
            '#{} - fitness: {}, solution: {}'.format(
                i + 1, func_vals[i], X[i].to_dict
            )
        )

    X = self.post_eval_check(X)
    self.data = self.data + X if hasattr(self, 'data') else X

    if self.data_file is not None:
        X.to_csv(self.data_file, header=False, append=True)

    self.fopt = self._get_best(self.data.fitness)
    _xopt = self.data[np.where(self.data.fitness == self.fopt)[0]]
    self.xopt = self._to_pheno(_xopt)
    if self._eval_type == 'dict':
        self.xopt = self.xopt[0]

    self._logger.info('fopt: {}'.format(self.fopt))
    self._logger.info('xopt: {}'.format(self.xopt))

    if not self.model.is_fitted:
        self._fBest_DoE = copy(self.fopt)  # the best point in the DoE
        self._xBest_DoE = copy(self.xopt)

    r2 = self.update_model()
    self._logger.info('Surrogate model r2: {}\n'.format(r2))

    if not warm_start:
        self.iter_count += 1
        self.hist_f.append(self.fopt)

####HYPEROPT####
import math
import random
#Fix bug of TPE: https://github.com/hyperopt/hyperopt/issues/768
def chooseRandomValueForParameter(self, parameter):
    if parameter.config.get("mode", "uniform") == "uniform":
        minVal = parameter.config["min"]
        maxVal = parameter.config["max"]

        if parameter.config.get("scaling", "linear") == "logarithmic":
            minVal = math.log(minVal)
            maxVal = math.log(maxVal)

        value = random.uniform(minVal, maxVal)

        if parameter.config.get("scaling", "linear") == "logarithmic":
            value = math.exp(value)

        if "rounding" in parameter.config:
            value = (
                    round(value / parameter.config["rounding"])
                    * parameter.config["rounding"]
            )
    elif parameter.config.get("mode", "uniform") == "normal":
        meanVal = parameter.config["mean"]
        stddevVal = parameter.config["stddev"]

        if parameter.config.get("scaling", "linear") == "logarithmic":
            meanVal = math.log(meanVal)
            stddevVal = math.log(stddevVal)

        value = random.gauss(meanVal, stddevVal)

        if parameter.config.get("scaling", "linear") == "logarithmic":
            value = math.exp(value)

        if "rounding" in parameter.config:
            value = (
                    round(value / parameter.config["rounding"])
                    * parameter.config["rounding"]
            )
    elif parameter.config.get("mode", "uniform") == "randint":
        min = parameter.config["min"]
        max = parameter.config["max"]-1
        value = random.randint(min, max)

    return value
