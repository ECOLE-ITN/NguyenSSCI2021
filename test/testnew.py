from BanditOpt.BO4ML import BO4ML, ConfigSpace, ConditionalSpace, \
    NominalSpace, OrdinalSpace, ContinuousSpace, Forbidden
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from BanditOpt.HyperoptConverter import OrginalToHyperopt
#from hyperopt import hp, tpe, atpe, STATUS_OK, Trials, fmin

def adapt_smo(datasetStr, randomstate, isNOSMO, Max_PCA_Component, min_Percent):
    cs = ConfigSpace()
    con = ConditionalSpace("test")
    dataset = NominalSpace([datasetStr], "dataset")
    alg_name = NominalSpace(['SVM', 'LinearSVC', 'RF', 'DTC', 'KNN', 'Quadratic'], 'alg_name')
    # dataset = NominalSpace( [datasetStr],"dataset")
    cs.add_multiparameter([dataset, alg_name])
    ##module1
    ####Missingvalue
    missingvalue = NominalSpace(['imputer'], 'missingvalue')
    strategy = NominalSpace(["mean", "median", "most_frequent", "constant"], 'strategy')
    ###ReScaling
    rescaling = NominalSpace(['MinMaxScaler', 'StandardScaler', 'RobustScaler'], 'rescaling')
    cs.add_multiparameter([missingvalue, strategy, rescaling])
    con.addConditional(strategy, missingvalue, ['imputer'])
    ####IMBALANCED
    random_state = NominalSpace([str(randomstate)], 'random_state')
    if (isNOSMO == True):
        imbalance = NominalSpace(['NONE'], 'imbalance')
        cs.add_multiparameter([random_state, imbalance])
    else:
        imbalance = NominalSpace(['NONE', 'SMOTE', 'SMOTENC', 'SMOTETomek', 'SMOTEENN'], 'imbalance')
        # categorical_features = NominalSpace(['True'],'categorical_features' )
        cs.add_multiparameter([random_state, imbalance])
        # con.addConditional(categorical_features, imbalance, ['SMOTENC'])
    # MODULE2

    ###PCA
    n_components_PCA = OrdinalSpace([2, Max_PCA_Component], 'n_components_PCA')
    svd_solver = NominalSpace(['auto', 'full', 'arpack', 'randomized'], 'svd_solver')
    copy = NominalSpace(['True', 'False'], 'copy')
    whiten_PCA = NominalSpace(['True', 'False'], 'whiten_PCA')
    iterated_power = OrdinalSpace([1, 50], 'iterated_power')
    tol_PCA = ContinuousSpace([1e-5, 1e-1], "tol_PCA")
    ###FastICA
    n_components = OrdinalSpace([2, 50], 'n_components')
    algorithm_FastICA = NominalSpace(['parallel', 'deflation'], 'algorithm_FastICA')
    fun = NominalSpace(['logcosh', 'exp', 'cube'], 'fun')
    whiten = NominalSpace(['True', 'False'], 'whiten')
    tol_FastICA = ContinuousSpace([1e-5, 1e-1], "tol_FastICA")
    ###SelectPercentile
    if (min_Percent < 100):
        FeaturePrepocessing = NominalSpace(['NONE', 'FastICA', 'PCA', 'SelectPercentile', ], 'FeaturePrepocessing')
        score_func = NominalSpace(['f_classif', 'f_regression', 'mutual_info_classif'], 'score_func')
        percentile = OrdinalSpace([min_Percent, 100], 'percentile')
        cs.add_multiparameter(
            [FeaturePrepocessing, n_components, algorithm_FastICA, whiten, fun, tol_FastICA, n_components_PCA,
             svd_solver, copy, whiten_PCA, iterated_power, tol_PCA, score_func, percentile])
        con.addMutilConditional([score_func, percentile], FeaturePrepocessing, 'SelectPercentile')
    else:
        FeaturePrepocessing = NominalSpace(['NONE', 'FastICA', 'PCA'], 'FeaturePrepocessing')
        cs.add_multiparameter([FeaturePrepocessing, n_components, algorithm_FastICA, whiten, fun, tol_FastICA,
                               n_components_PCA, svd_solver, copy, whiten_PCA, iterated_power, tol_PCA])
        # cs.add_multiparameter([FeaturePrepocessing,n_components,algorithm_FastICA,whiten,fun,tol_FastICA,n_components_PCA,svd_solver,copy,whiten_PCA,iterated_power,tol_PCA,score_func,percentile])
    con.addMutilConditional([n_components, algorithm_FastICA, whiten, fun, tol_FastICA], FeaturePrepocessing, 'FastICA')
    con.addMutilConditional([n_components_PCA, svd_solver, copy, whiten_PCA, iterated_power, tol_PCA],
                            FeaturePrepocessing, 'PCA')

    # MODULE3
    # elif (alg_nameStr == "RF"):
    n_estimators = OrdinalSpace([1, 1000], "n_estimators")
    criterion = NominalSpace(["gini", "entropy"], "criterion")
    max_depth = OrdinalSpace([2, 100], "max_depth")
    max_features = ContinuousSpace([0., 1.], "max_features")
    min_samples_split = OrdinalSpace([2, 20], "min_samples_split")
    min_samples_leaf = OrdinalSpace([1, 20], "min_samples_leaf")
    bootstrap = NominalSpace(['True', 'False'], "bootstrap")
    cs.add_multiparameter(
        [n_estimators, criterion, max_depth, max_features, min_samples_leaf, min_samples_split, bootstrap])
    con.addMutilConditional([n_estimators, criterion, max_depth, max_features, min_samples_leaf, min_samples_split,
                             bootstrap], alg_name, ['RF'])
    # elif (alg_nameStr == 'KNN'):
    n_neighbors = OrdinalSpace([1, 100], "n_neighbors")
    weights = NominalSpace(["uniform", "distance"], "weights")
    algorithm = NominalSpace(['auto', 'ball_tree', 'kd_tree', 'brute'], "algorithm")
    leaf_size = OrdinalSpace([1, 100], "leaf_size")
    p = OrdinalSpace([1, 2], "p")
    cs.add_multiparameter([n_neighbors, weights, algorithm, leaf_size, p])
    con.addMutilConditional([n_neighbors, weights, algorithm, leaf_size, p], alg_name, ['KNN'])
    # SVM
    probability = NominalSpace(['True', 'False'], 'probability')
    C = ContinuousSpace([0.03125, np.log(1e5)], 'C')
    kernel = NominalSpace(["rbf", "poly", "sigmoid"], 'kernel')
    coef0 = ContinuousSpace([-1, 1], 'coef0')
    degree = OrdinalSpace([2, 50], 'degree')
    shrinking = NominalSpace(['True', 'False'], "shrinking")
    gamma = ContinuousSpace([3.0517578125e-05, 8], "gamma")
    tol_SVM = ContinuousSpace([1e-5, 1e-1], 'tol_SVM')
    decision_function_shape = NominalSpace(["ovo", "ovr"], 'decision_function_shape')
    cs.add_multiparameter([probability, C, kernel, coef0, degree, shrinking, gamma, tol_SVM, decision_function_shape])
    con.addMutilConditional([probability, C, kernel, coef0, degree, shrinking, gamma, tol_SVM, decision_function_shape],
                            alg_name, ['SVM'])
    # 'name': 'LinearSVC',
    penalty = NominalSpace(
        ["hinge-l2-True", "squared_hinge-l2-True", "squared_hinge-l1-False", "squared_hinge-l2-False"], 'penalty')
    tol = ContinuousSpace([1e-5, 1e-1], 'tol')
    multi_class = NominalSpace(['ovr', 'crammer_singer'], 'multi_class')
    fit_intercept = NominalSpace(['True', 'False'], 'fit_intercept')
    intercept_scaling = ContinuousSpace([1e-5, 1e-1], 'intercept_scaling')
    C_Lin = ContinuousSpace([0.03125, np.log(1e5)], 'C_Lin')
    cs.add_multiparameter([penalty, tol, multi_class, fit_intercept, intercept_scaling, C_Lin])
    con.addMutilConditional([penalty, tol, multi_class, fit_intercept, intercept_scaling, C_Lin], alg_name,
                            ['LinearSVC'])

    # 'name': 'DTC',

    criterion_dtc = NominalSpace(["gini", "entropy"], 'criterion_dtc')
    max_depth_dtc = OrdinalSpace([2, 100], 'max_depth_dtc')
    max_features_dtc = NominalSpace(['auto', 'sqrt', 'log2', 'None'], 'max_features_dtc')
    min_samples_split_dtc = OrdinalSpace([2, 20], 'min_samples_split_dtc')
    min_samples_leaf_dtc = OrdinalSpace([1, 20], 'min_samples_leaf_dtc')
    splitter = NominalSpace(['best', 'random'], "splitter")
    # class_weight_dtc= NominalSpace(['balanced','None'],"class_weight_dtc", )
    # ccp_alpha = ContinuousSpace([0.0, 1.0],'ccp_alpha')
    cs.add_multiparameter([splitter, criterion_dtc, max_depth_dtc, max_features_dtc, min_samples_split_dtc,
                           min_samples_leaf_dtc])
    con.addMutilConditional([splitter, criterion_dtc, max_depth_dtc, max_features_dtc, min_samples_split_dtc,
                             min_samples_leaf_dtc], alg_name, ['DTC'])

    # metric = NominalSpace(['euclidean', 'manhattan', 'chebyshev', 'minkowski'], "metric")
    # p_sub_type = name
    ###'name': 'Quadratic',
    reg_param = ContinuousSpace([0.0, 50], 'reg_param')
    store_covariance = NominalSpace(['True', 'False'], 'store_covariance')
    tol_qua = ContinuousSpace([1e-5, 1e-1], 'tol_qua')
    cs.add_multiparameter([reg_param, store_covariance, tol_qua])
    con.addMutilConditional([reg_param, store_covariance, tol_qua], alg_name, ['Quadratic'])
    return cs, con
iris = datasets.load_iris()
X = iris.data
y = iris.target
def new_obj(params):
    print(params)
    FeaturePre = params['FeaturePrepocessing']
    FeaturePreName = FeaturePre['value']
    FeParams = {}
    loss = np.random.uniform(0.6, 1)
    if (FeaturePreName == 'SelectPercentile'):
        from sklearn.feature_selection import SelectPercentile, chi2, f_classif, f_regression, mutual_info_classif, \
            SelectKBest, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect
        score_func = FeaturePre['score_func']
        if (score_func == 'chi2'):
            score_func = chi2
        elif (score_func == 'f_classif'):
            score_func = f_classif
        elif (score_func == 'f_regression'):
            score_func = f_regression
        elif (score_func == 'mutual_info_classif'):
            score_func = mutual_info_classif
        FeParams['percentile'] = FeaturePre['percentile']
        transformer = SelectPercentile(score_func=score_func, **FeParams)
    if (FeaturePreName == 'SelectPercentile'):
        Xtr = transformer.fit_transform(X, y)
        loss=np.random.uniform(0,0.3)
    return loss
cs,con= adapt_smo('car', 18, False, 256, 0)
#opt = BO4ML(cs, new_obj, conditional=con,forbidden=None, max_eval=20, verbose=False, n_job=1, n_point=1,
 #           n_init_sample=3,SearchType="Bandit")
from Component.mHyperopt import tpe, rand, Trials,anneal, atpe
randomstate,dataset,HPOalg,method,SearchType=18,'car','hyperopt','atpe','Bandit'
trials= Trials
suggest = rand.suggest
#opt = BO4ML(cs, new_obj, forbidden=None, conditional=con, SearchType="Bandit",
   #         HPOopitmizer='hyperopt', max_eval=30,hpo_algo=suggest, hpo_show_progressbar=True)
opt = BO4ML(cs, new_obj, forbidden=None, conditional=con, SearchType=SearchType,minimize=True,
                HPOopitmizer=HPOalg, max_eval=500,hpo_trials=trials, hpo_show_progressbar=False,hpo_algo=suggest,random_seed=None)
xopt, fopt, _, eval_count = opt.run()
print(fopt)


#space=OrginalToHyperopt(search_space=cs,con=con)
#best = fmin(new_obj, space, algo=atpe.suggest, max_evals=100)