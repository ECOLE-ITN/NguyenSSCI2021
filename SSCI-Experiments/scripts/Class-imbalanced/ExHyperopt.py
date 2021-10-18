#!/usr/bin/env python
# coding: utf-8

# In[1]:
from functools import partial
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble  import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import pandas as pd
from pandas import Series, DataFrame
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.io.arff import loadarff
from scipy import interp
import json, logging, tempfile, sys, codecs, math, io, os,zipfile, arff, time, copy, csv,pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import roc_curve, auc
from hyperopt import fmin, tpe, hp, rand, STATUS_OK, Trials
from imblearn.over_sampling import (SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, SMOTENC,
                                    KMeansSMOTE, RandomOverSampler)
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
                                     NearMiss, TomekLinks,
                                     InstanceHardnessThreshold,
                                     CondensedNearestNeighbour,
                                     EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours,
                                     AllKNN,
                                     NeighbourhoodCleaningRule,
                                     OneSidedSelection)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.metrics import geometric_mean_score
from scipy.spatial import distance


# In[2]:
dataset=str(sys.argv[1])
seed=int(sys.argv[2])
HPOalg = 'TPE'
DataFolder='/home/nguyenda/DATA'
HomeFolder='/home/nguyenda/Anh'
file=DataFolder+'/ds/'+dataset+'/'+dataset+'.zip'
dataset=dataset+'.dat'
def getSP(seed):
    ##====SVM =====
    HPOspace = hp.choice('classifier_type', [
        {	'random_state': seed,
            'classifier': hp.choice('classifier',[
                {
                    'name': 'SVM',
                    'probability': hp.choice("probability", [True, False]),                
                    'C': hp.uniform('C', 0.03125 , 200 ),
                    'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),                
                    "degree": hp.randint("degree", 2,5),
                    "gamma": hp.choice('gamma',['auto','value','scale']),
                    'gamma_value': hp.uniform('gamma_value', 3.1E-05, 8),
                    "coef0": hp.uniform('coef0', -1, 1),
                    "shrinking": hp.choice("shrinking", [True, False]),
                    "tol": hp.uniform('tol_svm', 1e-05, 1e-01)#NEW
                },
                {
                    'name': 'RF',
                    'n_estimators': hp.randint("n_estimators", 1, 150),
                    'criterion': hp.choice('criterion', ["gini", "entropy"]),
                    'max_features': hp.choice('max_features_RF', [1, 'sqrt','log2',None]),               
                    'min_samples_split': hp.randint('min_samples_split', 2, 20),
                    'min_samples_leaf': hp.randint('min_samples_leaf', 1, 20),
                    'bootstrap': hp.choice('bootstrap',[True, False]),
                    'class_weight':hp.choice('class_weight',['balanced','balanced_subsample',None]),
                },
                {
                    'name': 'KNN',
                    'n_neighbors': hp.randint("n_neighbors_knn", 1, 51),
                    'weights': hp.choice('weights', ["uniform", "distance"]),
                    'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                    'p': hp.randint("p_value", 0, 20),
                },
                {
                    'name': 'DTC',
                    'criterion': hp.choice("criterion_dtc", ["gini", "entropy"]),
                    'max_features': hp.choice('max_features_dtc', [1, 'sqrt','log2',None]),
                    'max_depth': hp.choice('max_depth_dtc', range(2,20)),
                    'min_samples_split': hp.randint('min_samples_split_dtc', 2,20),
                    'min_samples_leaf': hp.randint('min_samples_leaf_dtc',1,20)     
                },
                {
                    'name': 'LR',
                    'C': hp.uniform('C_lr', 0.03125 , 100 ),
                    'penalty_solver': hp.choice("penalty_lr", ["l1+liblinear","l1+saga","l2+newton-cg","l2+lbfgs",
                                                               "l2+liblinear","l2+sag","l2+saga","elasticnet+saga",
                                                               "none+newton-cg","none+lbfgs","none+sag","none+saga"]),
                    'tol': hp.uniform('tol_lr', 1e-05, 1e-01),
                    'l1_ratio': hp.uniform('l1_ratio', 1e-09, 1)
                }]),

            'sub' : hp.choice('resampling_type',[
                {
                    'type':'NO'
                },
                #Over sampling
                {
                    'type': 'SMOTE',
                    'k_neighbors': hp.randint('k_neighbors_SMOTE',1,10),
                },
                {
                    'type': 'BorderlineSMOTE',
                    'k_neighbors': hp.randint('k_neighbors_Borderline',1,10),
                    'm_neighbors': hp.randint('m_neighbors_Borderline',1,10),
                    'kind' :  hp.choice('kind', ['borderline-1', 'borderline-2']),
                },
                {
                    'type': 'SMOTENC',
                    'categorical_features': True,
                    'k_neighbors': hp.randint('k_neighbors_SMOTENC',1,10), 
                },
                {
                    'type': 'SVMSMOTE',  
                    'k_neighbors': hp.randint('k_neighbors_SVMSMOTE',1,10), 
                    'm_neighbors': hp.randint('m_neighbors_SVMSMOTE',1,10),                
                    'out_step': hp.uniform('out_step', 0, 1)
                },
                {
                    'type': 'KMeansSMOTE',  
                    'k_neighbors': hp.randint('k_neighbors_KMeansSMOTE',1,10), 
                    'cluster_balance_threshold': hp.uniform('cluster_balance_threshold', 1e-2, 1), 
                },
                {
                    'type': 'ADASYN',
                    'n_neighbors' : hp.randint('n_neighbors_ADASYN',1,10),
                },    
                {
                    'type': 'RandomOverSampler'
                },
                #COMBINE RESAMPLING
                {
                    'type': 'SMOTEENN'
                },
                {
                    'type': 'SMOTETomek'
                },   
                 #UNDER RESAMPLING
                {
                    'type': 'CondensedNearestNeighbour',
                    'n_neighbors' : hp.randint('n_neighbors_CNN',1,50),
                    'n_seeds_S' : hp.randint('n_seeds_S_CNN',1,50),
                },
                {
                    'type': 'EditedNearestNeighbours',
                    'n_neighbors' : hp.randint('n_neighbors_ENN',1,20),
                    'kind_sel' : hp.choice('kind_sel_ENN',['all','mode']),
                },
                {
                    'type': 'RepeatedEditedNearestNeighbours',
                    'n_neighbors' : hp.randint('n_neighbors_RNN',1,20),
                    'kind_sel' : hp.choice('kind_sel_RNN',['all','mode']),
                },
                {
                    'type': 'AllKNN',
                    'n_neighbors' : hp.randint('n_neighbors_AKNN',1,20),
                    'kind_sel' : hp.choice('kind_sel_AKNN',['all','mode']),
                    'allow_minority' : hp.choice('allow_minority_AKNN', [True, False])
                },
                {
                    'type': 'InstanceHardnessThreshold',
                    'estimator': hp.choice('estimator_IHTh', ['knn', 'decision-tree', 'adaboost','gradient-boosting','linear-svm', None]),
                    'cv' : hp.randint('cv_IHTh',2,10,)                
                },
                {
                    'type': 'NearMiss',
                    'version' : hp.choice('version_NM',[1,2,3]),
                    'n_neighbors' : hp.randint('n_neighbors_NM',1,20),
                    'n_neighbors_ver3' : hp.randint('n_neighbors_ver3_NM',1,20)              
                },
                {
                    'type': 'NeighbourhoodCleaningRule',
                    'n_neighbors' : hp.randint('n_neighbors_NCR',1,20),
                    'threshold_cleaning' : hp.uniform('threshold_cleaning_NCR',0,1)
                },
                {
                    'type': 'OneSidedSelection',
                    'n_neighbors' : hp.randint('n_neighbors_OSS',1,20),
                    'n_seeds_S' : hp.randint('n_seeds_S_OSS',1,20)
                },
                {
                    'type': 'RandomUnderSampler',
                    'replacement' : hp.choice('replacement_RUS', [True, False])                
                },
                {
                    'type': 'TomekLinks'
                },
                {
                    'type': 'ClusterCentroids',
                    'estimator': hp.choice('estimator_CL',['KMeans', 'MiniBatchKMeans']),
                    'voting' : hp.choice('voting_CL',['hard', 'soft'])
                }


            ])
        }
    ])

    return HPOspace
# In[4]:
resampler_group={'NO':'NO','SMOTE':'OVER','BorderlineSMOTE':'OVER','SMOTENC':'OVER','SVMSMOTE':'OVER','KMeansSMOTE':'OVER'
                 ,'ADASYN':'OVER','RandomOverSampler':'OVER',
                 'SMOTEENN':'COMBINE','SMOTETomek':'COMBINE',
                 'CondensedNearestNeighbour':'UNDER','EditedNearestNeighbours':'UNDER',
                 'RepeatedEditedNearestNeighbours':'UNDER','AllKNN':'UNDER',
                 'InstanceHardnessThreshold':'UNDER','NearMiss':'UNDER',
                            'NeighbourhoodCleaningRule':'UNDER','OneSidedSelection':'UNDER','RandomUnderSampler':'UNDER',
                            'TomekLinks':'UNDER','ClusterCentroids':'UNDER'}

def fscore(params_org):
    #print(params_org)
    parambk = copy.deepcopy(params_org)
    ifError =0
    global best, HPOalg,params_best, errorcount,resampler_group,randomstate
    params= params_org['classifier']
    classifier = params.pop('name')
    xxx = params_org.pop('random_state')
    p_random_state=randomstate
    if (classifier == 'SVM'):  
        param_value= params.pop('gamma_value')
        if(params['gamma'] == "value"):
            params['gamma'] = param_value
        else:
            pass   
        clf = SVC(max_iter = 10000, cache_size= 700, random_state = p_random_state,**params)
        #max_iter=10000 and cache_size= 700 https://github.com/EpistasisLab/pennai/issues/223
        #maxvalue https://github.com/hyperopt/hyperopt-sklearn/blob/fd718c44fc440bd6e2718ec1442b1af58cafcb18/hpsklearn/components.py#L262
    elif(classifier == 'RF'):        
        clf = RandomForestClassifier(random_state = p_random_state, **params)
    elif(classifier == 'KNN'):
        p_value = params.pop('p')
        if(p_value==0):
            params['metric'] = "chebyshev"
        elif(p_value==1):
            params['metric'] = "manhattan"
        elif(p_value==2):
            params['metric'] = "euclidean"
        else:
            params['metric'] = "minkowski"
            params['p'] = p_value
        #https://github.com/hyperopt/hyperopt-sklearn/blob/fd718c44fc440bd6e2718ec1442b1af58cafcb18/hpsklearn/components.py#L302
        clf = KNeighborsClassifier(**params)
    elif(classifier == 'DTC'):        
        clf = DecisionTreeClassifier(random_state = p_random_state, **params)
    elif(classifier == 'LR'):        
        penalty_solver = params.pop('penalty_solver')
        params['penalty'] = penalty_solver.split("+")[0]
        params['solver'] = penalty_solver.split("+")[1]
        clf = LogisticRegression(random_state = p_random_state, **params)
    #resampling parameter
    p_sub_params= params_org.pop('sub')
    p_sub_type = p_sub_params.pop('type')
    #sampler = p_sub_params.pop('smo_grp')
    sampler = resampler_group[p_sub_type]
    if p_sub_type not in ('EditedNearestNeighbours','RepeatedEditedNearestNeighbours','AllKNN',
                              'NearMiss','NeighbourhoodCleaningRule','TomekLinks'):
            p_sub_params['random_state']=p_random_state

    gmean = []
    if 'n_neighbors' in p_sub_params:
        p_sub_params['n_neighbors']=int(p_sub_params['n_neighbors'])
    if (p_sub_type == 'SMOTE'):
        smo = SMOTE(**p_sub_params)
    elif (p_sub_type == 'ADASYN'):
        smo = ADASYN(**p_sub_params)
    elif (p_sub_type == 'BorderlineSMOTE'):
        smo = BorderlineSMOTE(**p_sub_params)
    elif (p_sub_type == 'SVMSMOTE'):
        smo = SVMSMOTE(**p_sub_params)
    elif (p_sub_type == 'SMOTENC'):
        smo = SMOTENC(**p_sub_params)
    elif (p_sub_type == 'KMeansSMOTE'):
        smo = KMeansSMOTE(**p_sub_params)
    elif (p_sub_type == 'RandomOverSampler'):
        smo = RandomOverSampler(**p_sub_params)
#Undersampling
    elif (p_sub_type == 'TomekLinks'):
        smo = TomekLinks(**p_sub_params)
    elif (p_sub_type == 'ClusterCentroids'):
        if(p_sub_params['estimator']=='KMeans'):
            p_sub_params['estimator']= KMeans(random_state = p_random_state)
        elif(p_sub_params['estimator']=='MiniBatchKMeans'):
            p_sub_params['estimator']= MiniBatchKMeans(random_state = p_random_state)
        smo = ClusterCentroids(**p_sub_params) 
    elif (p_sub_type == 'RandomUnderSampler'):
        smo = RandomUnderSampler(**p_sub_params)
    elif (p_sub_type == 'NearMiss'):
        smo = NearMiss(**p_sub_params)
    elif (p_sub_type == 'InstanceHardnessThreshold'):
        if(p_sub_params['estimator']=='knn'):
            p_sub_params['estimator']= KNeighborsClassifier()
        elif(p_sub_params['estimator']=='decision-tree'):
            p_sub_params['estimator']=DecisionTreeClassifier()
        elif(p_sub_params['estimator']=='adaboost'):
            p_sub_params['estimator']=AdaBoostClassifier()
        elif(p_sub_params['estimator']=='gradient-boosting'):
            p_sub_params['estimator']=GradientBoostingClassifier()
        elif(p_sub_params['estimator']=='linear-svm'):
            p_sub_params['estimator']=CalibratedClassifierCV(LinearSVC())
        elif(p_sub_params['estimator']=='random-forest'):
            p_sub_params['estimator']=RandomForestClassifier(n_estimators=100)
        smo = InstanceHardnessThreshold(**p_sub_params) 
    elif (p_sub_type == 'CondensedNearestNeighbour'):
        smo = CondensedNearestNeighbour(**p_sub_params)
    elif (p_sub_type == 'EditedNearestNeighbours'):
        smo = EditedNearestNeighbours(**p_sub_params)
    elif (p_sub_type == 'RepeatedEditedNearestNeighbours'):
        smo = RepeatedEditedNearestNeighbours(**p_sub_params) 
    elif (p_sub_type == 'AllKNN'):
        smo = AllKNN(**p_sub_params)
    elif (p_sub_type == 'NeighbourhoodCleaningRule'):
        smo = NeighbourhoodCleaningRule(**p_sub_params) 
    elif (p_sub_type == 'OneSidedSelection'):
        smo = OneSidedSelection(**p_sub_params)
#Combine
    elif (p_sub_type == 'SMOTEENN'):
        smo = SMOTEENN(**p_sub_params)
    elif (p_sub_type == 'SMOTETomek'):
        smo = SMOTETomek(**p_sub_params)
    e=''
    
    try:        
        for train, test in cv.split(X, y):
            if(p_sub_type=='NO'):
                X_smo_train, y_smo_train = X[train], y[train]
            else:
                X_smo_train, y_smo_train = smo.fit_resample(X[train], y[train])
            y_test_pred = clf.fit(X_smo_train, y_smo_train).predict(X[test])
            gm = geometric_mean_score(y[test], y_test_pred, average='binary')
            gmean.append(gm)
        mean_g=np.mean(gmean)
    except Exception as eec:
        #print(parambk,eec)
        e=eec
        mean_g = 0
        ifError =1 
        errorcount = errorcount+1
    gm_loss = 1 - mean_g
    abc=time.time()-starttime
    if mean_g > best:
        best = mean_g
        params_best = copy.deepcopy(parambk)
    return {'loss': gm_loss,
            'mean': mean_g,
            'status': STATUS_OK,         
            # -- store other results like this
            'run_time': abc,
            'iter': iid,
            'current_best': best,
            'eval_time': time.time(),
            'classifier': classifier,
            'SamplingGrp': sampler,
            'SamplingType': p_sub_type,
            'ifError': ifError,
            'Error': e,
            'params' : parambk,
            'attachments':
                {'time_module': pickle.dumps(time.time)}
           }   

    

zf = zipfile.ZipFile(file) 
in_mem_fo = io.TextIOWrapper(io.BytesIO(zf.read(dataset)), encoding='utf-8')
txt=''

for i in in_mem_fo:
    if("@input" not in i and '@output' not in i ):
        newi=''
        for e in i.split(','):
            if(newi==''):
                newi=e.strip()
            else:
                newi=newi+','+e.strip()
        txt=txt+newi+'\n'
abc=io.StringIO(txt)
data,meta=loadarff(abc)
data=pd.DataFrame(data)
enc = LabelEncoder()
#######LOAD TRAIN DATA######
for a in [ col  for col, data in data.dtypes.items() if data == object]:
    #print(a)
    data[a] = data[a].str.decode('utf-8') 
    try:
        data[a]=data[a].astype('int64')
        #print(a)
    except:
        data[a] = enc.fit_transform(data[a])
X= data[data.columns[:-1]].to_numpy()
y= data[data.columns[-1]].to_numpy()
X = StandardScaler().fit_transform(X)
X = np.c_[X]
k = 5        
cv = StratifiedKFold(n_splits=k)
# In[5]:
seeds= [seed] if seed>0 else [*range(1,11)] 
for n_init_sample in [20,50,100]:
    for randomstate in seeds: 
        print('\033[91m',HPOalg,'==',randomstate,'=== START DATASET: ', dataset, '=======', '\033[0m') 
        space = getSP(randomstate)   
        best,params_best = 0,''
        trials = Trials()
        starttime = time.time()
        ran_best = 0 
        best = 0        
        iid = 0
        errorcount=0
        rstate=np.random.RandomState(randomstate)
        suggest= partial(tpe.suggest, n_startup_jobs=n_init_sample)
        try:
            xOpt= fmin(fscore, space, algo=suggest, max_evals=500, trials=trials, rstate=rstate)
        except:
            print('==ERROR: RANDOM-',dataset,'===')
        runtime=time.time()-starttime

        try:
            ran_results = pd.DataFrame({'current_best': [x['current_best'] for x in trials.results],
                                        'run_time':[x['run_time'] for x in trials.results],
                                        'classifier': [x['classifier'] for x in trials.results],
                                        'SamplingGrp': [x['SamplingGrp'] for x in trials.results],                                    
                                        'SamplingType': [x['SamplingType'] for x in trials.results], 
                                        'ifError': [x['ifError'] for x in trials.results], 
                                        'Error': [x['Error'] for x in trials.results], 
                                        'loss': [x['loss'] for x in trials.results], 
                                        'mean': [x['mean'] for x in trials.results], 
                                        'iteration': trials.idxs_vals[0]['classifier_type'],
                                        'params':[x['params'] for x in trials.results]})
            ran_results.to_csv(DataFolder+'/TPEHyperopt/'+str(n_init_sample)+'_hyperopt_'+HPOalg+'_'+dataset+'_'+str(randomstate)+'.csv', 
                               index = True, header=True)
        except:
            print('ERROR: No logfile')
        finallog= HomeFolder+"/TPE_hyperopt_finallog.csv"
        if (os.path.exists(finallog)==False):
            with open(finallog, "a") as f:    
                wr = csv.writer(f, dialect='excel')
                wr.writerow(['dataname','HPOalg','random_state','initsample','mean', 'params','runtime','errorcount'])
        with open(finallog, "a") as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow([dataset,HPOalg,randomstate,n_init_sample,best,params_best,runtime,errorcount])

