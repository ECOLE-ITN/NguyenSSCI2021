from sklearn import datasets
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from BanditOpt import BO4ML, ConditionalSpace, ConfigSpace, ContinuousSpace, NominalSpace, OrdinalSpace, Forbidden
from Component.mHyperopt import tpe, rand, Trials
import warnings
warnings.filterwarnings("ignore")
# define Configuration space
search_space = ConfigSpace()

# Define Search Space
alg_namestr = NominalSpace(["SVM", "RF"], "alg_namestr")

# Define Search Space for Support Vector Machine
kernel = NominalSpace(["linear", "rbf"], "kernel")
test=NominalSpace(["A","B"],"test")
C = ContinuousSpace([1e-2, 100], "C")
degree = OrdinalSpace([1, 5], 'degree')
coef0 = ContinuousSpace([0.0, 10.0], 'coef0')
gamma = ContinuousSpace([0, 20], 'gamma')
# Define Search Space for Random Forest
n_estimators = OrdinalSpace([5, 100], "n_estimators")
criterion = NominalSpace(["gini", "entropy"], "criterion")
max_depth = OrdinalSpace([10, 200], "max_depth")
max_features = NominalSpace(['auto', 'sqrt', 'log2'], "max_features")
alone = NominalSpace(['A1', 'A2', 'A3'], "alone")
# Add Search space to Configuraion Space
search_space.add_multiparameter([alg_namestr, kernel, C, degree, coef0, gamma
                                    , n_estimators, criterion, max_depth, max_features, test,alone])
# Define conditional Space
con = ConditionalSpace("conditional")
con.addMutilConditional([kernel, C, degree, coef0, test], alg_namestr, ["SVM"])
con.addMutilConditional([n_estimators, criterion, max_depth, max_features], alg_namestr, ["RF"])
con.addConditional(gamma,test,'A')
fobr = Forbidden()
fobr.addForbidden(max_features, "auto", criterion, "gini")
fobr.addForbidden(test,"A",kernel,"linear")
iris = datasets.load_iris()
X = iris.data
y = iris.target

def new_obj(params):
    print(params)
    return (np.random.uniform(0,1))
def obj_func(params):
    params = {k: params[k] for k in params if params[k]}
    # print(params)
    classifier = params['alg_namestr']
    params.pop("alg_namestr", None)
    params.pop("test",None)
    # print(params)
    clf = SVC()
    if (classifier == 'SVM'):
        clf = SVC(**params)
    elif (classifier == 'RF'):
        clf = RandomForestClassifier(**params)
    mean = cross_val_score(clf, X, y).mean()
    loss = 1 - mean
    # print (mean)
    return loss
opt = BO4ML(search_space, new_obj,forbidden=fobr,conditional=con,SearchType="Bandit", max_eval=50)
suggest = tpe.suggest
#opt = BO4ML(search_space, new_obj, forbidden=None, conditional=None, SearchType="Bandit",
    #        HPOopitmizer='hyperopt', max_eval=30,hpo_algo=suggest, hpo_show_progressbar=True)
xopt, fopt, _, eval_count = opt.run()
print(xopt,fopt)