from __future__ import absolute_import
from hyperopt import hp, tpe, atpe, STATUS_OK, Trials, fmin
#from Component.mHyperopt import tpe, rand, Trials,anneal, atpe
import numpy as np
randomstate=18
def get_SearchSpace(dataset, isNOSMO, Max_PCA_Component, min_percent):
    if (isNOSMO == True):
        imbalance = 'NONE'  # update
    else:
        imbalance = hp.choice('imbalance', [{
            'name': 'NONE'},
            {'name': 'SMOTE'},
            {'name': 'SMOTENC',
             'categorical_features': hp.choice('categorical_features', [True])
             },
            {'name': 'SMOTETomek'},
            {'name': 'SMOTEENN'}])
        # imbalance= ['NONE','SMOTE','SMOTENC','SMOTETomek','SMOTEENN'] #update
    HPOspace = hp.choice('classifier_type', [
        {
            'module1': hp.choice('DataPrepocessing', [{
                'missingvalue': hp.choice('missing', [{
                    'name': 'imputer',
                    'strategy': hp.choice('strategy', ["mean", "median", "most_frequent", "constant"])
                }]),
                'dataset': dataset,
                'imbalance': imbalance,  # update
                'rescaling': hp.choice('rescaling', ['NONE', 'MinMaxScaler', 'StandardScaler', 'RobustScaler']),
            }]),
            'module2': hp.choice('FeaturePrepocessing', [  # NEW
                {
                    'name': 'NONE'
                },
                {
                    'name': 'FastICA',
                    'n_components': hp.choice('n_components', range(2, 50)),
                    'algorithm_FastICA': hp.choice('algorithm_FastICA', ['parallel', 'deflation']),
                    'whiten': hp.choice('whiten', [True, False]),
                    'fun': hp.choice('fun', ['logcosh', 'exp', 'cube']),
                    'tol_FastICA': hp.uniform('tol_FastICA', 1e-5, 1e-1),
                },
                {
                    'name': 'PCA',
                    'n_components_PCA': hp.choice('n_components_PCA', range(2, Max_PCA_Component)),
                    'svd_solver': hp.choice('svd_solver', ['auto', 'full', 'arpack', 'randomized']),
                    'copy': hp.choice('copy', [True, False]),
                    'whiten_PCA': hp.choice('whiten_PCA', [True, False]),
                    'iterated_power': hp.choice("iterated_power", range(1, 50)),
                    'tol_PCA': hp.uniform('tol_PCA', 1e-5, 1e-1),
                },
                {
                    'name': 'SelectPercentile',
                    'score_func': hp.choice('score_func', ['f_classif', 'f_regression', 'mutual_info_classif']),
                    'percentile': hp.choice("percentile", range(min_percent, 100)),
                }
            ]),
            'module3': hp.choice('Classifier', [

                {
                    'name': 'RF',
                    'n_estimators': hp.choice("n_estimators", range(1, 1000)),
                    'criterion': hp.choice('criterion', ["gini", "entropy"]),
                    'max_depth': hp.choice("max_depth", range(2, 100)),  # update
                    'max_features': hp.uniform('max_features', 0., 1.),  # update
                    'min_samples_split': hp.choice('min_samples_split', range(2, 20)),
                    'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 20)),
                    'bootstrap': hp.choice('bootstrap', [True, False]),
                    'random_state': randomstate
                },
                {
                    'name': 'KNN',
                    'n_neighbors': hp.choice("n_neighbors", range(1, 100)),  # update
                    'weights': hp.choice('weights', ["uniform", "distance"]),  # new
                    'algorithm': hp.choice('algorithm', ["auto", "ball_tree", "kd_tree", "brute"]),  # new
                    'leaf_size': hp.choice("leaf_size", range(1, 100)),  # new
                    'p': hp.choice('p', range(1, 2))
                },
                {
                    'name': 'SVM',
                    'probability': hp.choice('probability', [True, False]),
                    'random_state': randomstate,
                    'C': hp.uniform('C', 0.3125, np.log(1e5)),
                    'kernel': hp.choice('kernel', ["rbf", "poly", "sigmoid"]),  # update
                    "coef0": hp.uniform('coef0', -1, 1),
                    "degree": hp.choice("degree", range(2, 50)),  # update
                    "shrinking": hp.choice("shrinking", [True, False]),
                    "gamma": hp.uniform('gamma', 3.0517578125e-05, 8),
                    'tol': hp.uniform('tol_SVM', 1e-5, 1e-1),
                    'decision_function_shape': hp.choice('decision_function_shape', ["ovo", "ovr"])  # new
                },
                {
                    'name': 'LinearSVC',
                    'penalty': hp.choice('penalty', ["hinge-l2-True", "squared_hinge-l2-True", "squared_hinge-l1-False",
                                                     "squared_hinge-l2-False"]),
                    # "loss" : hp.choice('loss',["hinge","squared_hinge"]),
                    # 'dual' : hp.choice('dual', [True, False]), #update
                    # 'dual' : False,
                    'tol': hp.uniform('tol', 1e-5, 1e-1),
                    'multi_class': hp.choice('multi_class', ['ovr', 'crammer_singer']),
                    'fit_intercept': hp.choice('fit_intercept', [True, False]),  # update
                    'intercept_scaling': hp.uniform('intercept_scaling', 1e-5, 1e-1),
                    'random_state': randomstate,
                    'C': hp.uniform('C_Lin', 0.03125, np.log(1e5))  # update
                },
                {
                    'name': 'DTC',
                    'criterion': hp.choice('criterion_dtc', ["gini", "entropy"]),
                    'max_depth': hp.choice('max_depth_dtc', range(2, 100)),  # update
                    'max_features': hp.choice('max_features_dtc', ['auto', 'sqrt', 'log2', None]),
                    'min_samples_split': hp.choice('min_samples_split_dtc', range(2, 20)),
                    'min_samples_leaf': hp.choice('min_samples_leaf_dtc', range(1, 20)),
                    # 'ccp_alpha': hp.uniform('ccp_alpha', 0.0, 5.0),
                    'splitter': hp.choice('splitter', ['best', 'random']),  # new
                    'random_state': randomstate
                },
                {
                    'name': 'Quadratic',
                    'reg_param': hp.uniform('reg_param', 0.0, 50.0),  # update
                    'store_covariance': hp.choice('store_covariance', [True, False]),  # new
                    'tol': hp.uniform('tol_qua', 1e-5, 1e-1)  # new
                }]),
        }, ])
    return HPOspace

space= get_SearchSpace('car', False, 6, 33)
def new_obj(params):
    #print(params)
    loss=np.random.uniform(0,1)
    return {"loss":loss,
            'status': STATUS_OK}
trials=Trials()
best = fmin(new_obj, space, algo=atpe.suggest, max_evals=500,trials=trials)