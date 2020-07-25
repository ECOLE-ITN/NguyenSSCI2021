from BanditOpt.BO4ML import BO4ML, ConfigSpace, ConditionalSpace, NominalSpace, OrdinalSpace, ContinuousSpace, Forbidden
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
search_space = ConfigSpace()
# Define Search Space
abc = NominalSpace(['A','B'],'ABC')
alg_namestr = NominalSpace(["SVM", "RF"], "alg_namestr")

# Define Search Space for Support Vector Machine
kernel = NominalSpace(["linear", "rbf", "poly", "sigmoid"], "kernel")
C = ContinuousSpace([1e-2, 100], "C")
degree = OrdinalSpace([1, 5], 'degree')
coef0 = ContinuousSpace([0.0, 10.0], 'coef0')
gamma = ContinuousSpace([0, 20], 'gamma')
# Define Search Space for Random Forest
n_estimators = OrdinalSpace([5, 100], "n_estimators")
criterion = NominalSpace(["gini", "entropy"], "criterion")
max_depth = OrdinalSpace([10, 200], "max_depth")
max_features = NominalSpace(['auto', 'sqrt', 'log2'], "max_features")

# Add Search space to Configuraion Space
search_space.add_multiparameter([alg_namestr, kernel, C, degree, coef0, gamma
                                    , n_estimators, criterion, max_depth, max_features])
# Define conditional Space
con = ConditionalSpace("conditional")
con.addMutilConditional([kernel, C, degree, coef0, gamma], alg_namestr, "SVM")
con.addMutilConditional([n_estimators, criterion, max_depth, max_features], alg_namestr, ["RF"])
forb = Forbidden()
forb.addForbidden(abc,"A",alg_namestr,"SVM")
iris = datasets.load_iris()
X = iris.data
y = iris.target


def new_obj(params):
    print(params)
    return (np.random.uniform(0, 1))
def obj_func(params):
    params = {k: params[k] for k in params if params[k]}
    # print(params)
    classifier = params['alg_namestr']
    params.pop("alg_namestr", None)
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


opt = BO4ML(search_space, new_obj, conditional=con,forbidden=forb, max_eval=20, verbose=True, n_job=1, n_point=1,
            n_init_sample=3,SearchType="BO")

xopt, fopt, _, eval_count = opt.run()
print(fopt)
