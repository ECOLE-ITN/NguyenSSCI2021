[![Actions Status](https://api.travis-ci.org/anh05/BO4ML.svg?branch=master)](http://travis-ci.org/anh05/BO4ML)
# Hyperparameter optimization for Machine learning 

Copyright (C) 2019-2021 [ECOLE Project](https://ecole-itn.eu/), [NACO Group](https://naco.liacs.nl/)

### Contact us

Mr. Duc Anh Nguyen

Email:d-dot-a-dot-nguyen-at-liacs-dot-leidenuniv-dot-nl

[Project website](http://hyperparameter.ml)
## Installation
### Requirements

As requirements  mentioned in `requirements.txt`, this package requires [bayes-optim](https://github.com/wangronin/Bayesian-Optimization) , 
and [hyperopt](https://github.com/hyperopt/hyperopt) as build dependencies:

```shell
pip install bayes-optim
pip install hyperopt
```
### Installation

You could either install the stable version on `pypi`:

```shell
pip install BO4ML
```

Or, take the lastest version from github:

```shell
pip install git+https://github.com/anh05/BO4ML.git
```
--
```shell
git clone https://github.com/anh05/BO4ML.git
cd BO4ML && python setup.py install --user
```

## Example
Define a Seach space
```python
from BanditOpt.BO4ML import ConfigSpace, ConditionalSpace, NominalSpace, OrdinalSpace, ContinuousSpace, Forbidden

search_space = ConfigSpace()
# Define Search Space
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
                                    , n_estimators, criterion, max_depth, max_features,abc])
# Define conditional Space
con = ConditionalSpace("conditional")
con.addMutilConditional([kernel, C, degree, coef0, gamma], alg_namestr, "SVM")
con.addMutilConditional([n_estimators, criterion, max_depth, max_features], alg_namestr, ["RF"])
# Define infeasible space (if any)
#forb = Forbidden()
#forb.addForbidden(abc,["A","C","D"],alg_namestr,"SVM")
```
Load iris data
```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
```
Define an objective function which returns a real-value
```python
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
def obj_func(params):
    params = {k: params[k] for k in params if params[k]}
    classifier = params['alg_namestr']
    params.pop("alg_namestr", None)    
    if (classifier == 'SVM'):
        clf = SVC(**params)
    elif (classifier == 'RF'):
        clf = RandomForestClassifier(**params)
    mean = cross_val_score(clf, X, y).mean()
    loss = 1 - mean
    return loss
```
Optimize with [bayes-optim](https://github.com/wangronin/Bayesian-Optimization):
```python
from BanditOpt.BO4ML import BO4ML
opt = BO4ML(search_space, obj_func, 
            conditional=con, #conditional 
            #forbidden=forb, #No infeasible space defined in this example
            max_eval=20, #number of evaluations
            n_init_sample=3, #number of init sample 
            SearchType="Bandit"# SearchType="Bandit"# "Bandit" to use our divide and conquer approach. Otherwise, the original library to be used
            )                
```
Optimize with [bayes-optim](https://github.com/wangronin/Bayesian-Optimization):
```python
from BanditOpt.BO4ML import BO4ML

opt = BO4ML(search_space, obj_func, 
            conditional=con, #conditional 
            HPOopitmizer='bayes-optim', #use bayes-optim
            #forbidden=forb, #No infeasible space defined in this example
            max_eval=50, #number of evaluations
            n_init_sample=3, #number of init sample 
            SearchType="Bandit"# set "Bandit" to use our divide and conquer approach. Otherwise, the original library to be used
            )        
xopt, fopt, _, eval_count = opt.run()
print(xopt,fopt)        
```
Optimize with [hyperopt](https://github.com/hyperopt/hyperopt)
```python
from Component.mHyperopt import tpe, rand, atpe, anneal 
from BanditOpt.BO4ML import BO4ML
opt = BO4ML(search_space, new_obj, 
            conditional=con, #conditional 
            #forbidden=forb, #No infeasible space defined in this example SearchType="NoBandit",
            HPOopitmizer='hyperopt', #use hyperopt
            max_eval=50, #number of evaluations
            n_init_sample=3, #number of init sample 
            hpo_algo=tpe.suggest, #tpe, rand, atpe, anneal
            SearchType="Bandit"# set "Bandit" to use our divide and conquer approach. Otherwise, the original library to be used
            )
xopt, fopt, listofTrial, eval_count = opt.run()
print(xopt,fopt)
#listofTrial: see hyperopt document for ``trails''
```
#Reference
to be update

## Acknowledgment

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 766186 (ECOLE).