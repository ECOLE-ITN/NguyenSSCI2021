from BayesOpt.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace
import numpy as np
import copy
from BayesOpt.BayesOpt import BO
from BayesOpt.base import Solution


###Search space
def init_SearchSpace(self, bounds, var_name, name, default=None):
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
    self.var_type = None
    self.levels = None
    self.precision = None

    if var_name is not None:
        if isinstance(var_name, str):
            if self.dim > 1:
                var_name = [var_name + '_' + str(_) for _ in range(self.dim)]
            else:
                var_name = [var_name]
        assert len(var_name) == self.dim
        self.var_name = var_name


def init_NominalSpace(self, levels, var_name='d', name=None, default=None):
    levels = np.atleast_2d(levels)
    levels = [np.unique(l).tolist() for l in levels]

    super(NominalSpace, self).__init__(levels, var_name, name, default)
    self.var_type = ['N'] * self.dim
    self._levels = [np.array(b) for b in self.bounds]
    self._set_index()
    self._set_levels()


def init_ContinuousSpace(self, bounds, var_name='r', name=None, precision=None, default=None):
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

    assert all(self._bounds[0, :] < self._bounds[1, :])


def init_OrdinalSpace(self, bounds, var_name='i', name=None, default = None):
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
        # hyperparameter._n_levels = [len(l) for l in hyperparameter._levels]
    elif (isinstance(hyperparameter, ContinuousSpace)):
        pass
    return hyperparameter


def imputation(conditional, x, var_names, defaultvalue):
    lsParentName, childList, lsFinalSP,ActiveLst, noCheckForb = [], [], [], [], []
    for i, con in conditional.conditional.items():
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
                    del nodeChilds[idx]
                if (len(childofChild) > 0):
                    nodeChilds = childofChild
    noCheckForb =[x for x in var_names if x not in ActiveLst]
    for node in noCheckForb:
        x[var_names.index(node)] = defaultvalue[node]

    return x, noCheckForb


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
            noChecklst=[]
            if (isBandit == False and self._conditional != None):
                x, noChecklst = imputation(self._conditional, x, self.var_names, defaultvalue)
            else:
                pass
            ##Check Forbidden
            #noCheckid=[i for (i, v) in enumerate(self.var_names) if v in noChecklst]
            x_dict = dict(zip(self.var_names, x))
            x_dict= dict((i, v) for (i, v) in x_dict.items() if i not in noChecklst)
            isFOB = False
            if(self._forbidden!=None):
                for fname, fvalue in self._forbidden.forbList.items():
                    hp_left = [(key, value) for (key, value) in x_dict.items() if
                               key == fvalue.left and len(set([value]).intersection(fvalue.leftvalue)) > 0]
                    hp_right = [(key, value) for (key, value) in x_dict.items() if
                                key == fvalue.right and len(set([value]).intersection(fvalue.rightvalue)) > 0]
                    hp_add1,hp_add2 =[],[]
                    if(fvalue.ladd1 != None):
                        hp_add1= [(key, value) for (key, value) in x_dict.items() if
                               key == fvalue.ladd1 and len(set([value]).intersection(fvalue.ladd1value)) > 0]
                    if (fvalue.ladd2 != None):
                        hp_add2 = [(key, value) for (key, value) in x_dict.items() if
                                   key == fvalue.ladd2 and len(set([value]).intersection(fvalue.ladd2value)) > 0]
                    if (fvalue.ladd1!=None and fvalue.ladd2 != None):
                        if (len(hp_left) > 0 and len(hp_right) > 0 and len(hp_add1)>0 and len(hp_add2)>0):
                            isFOB = True
                    elif(fvalue.ladd1!=None):
                        if (len(hp_left) > 0 and len(hp_right) > 0 and len(hp_add1)>0):
                            isFOB = True
                    else:
                        if (len(hp_left) > 0 and len(hp_right) > 0):
                            isFOB = True

            """d.a.nguyen: update X based on conditional ::: end"""
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
