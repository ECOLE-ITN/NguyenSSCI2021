#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from hyperopt import fmin, tpe, hp, rand, STATUS_OK, Trials, STATUS_FAIL
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
from BanditOpt import ConfigSpace, ConditionalSpace, AlgorithmChoice, IntegerParam, FloatParam, CategoricalParam, Forbidden
from BanditOpt.BO4ML import BO4ML




# In[3]:
dataset=str(sys.argv[1])
seed=int(sys.argv[2])
HPOalg = 'TPE'
DataFolder='/home/nguyenda/DATA'
HomeFolder='/home/nguyenda/Anh'
file=DataFolder+'/ds/'+dataset+'/'+dataset+'.zip'
dataset=dataset+'.dat'
class Exsupport():
    errorcount=0
    def __init__(self,file, dataset, seed, initsample):
        self.file=file
        Exsupport.errorcount=0
        self.dataset=dataset
        self.randomseed=seed
        self.initsample=initsample
        self.initcomb=[]
        self.init_eval=dict()
        self.cv = StratifiedKFold(n_splits=5)
        self.best,self.params_best = 0,''
        self._idxi=0
        self.X,self.y=self.getdata()
        self.resampler_group={'NO':'NO','SMOTE':'OVER','BorderlineSMOTE':'OVER','SMOTENC':'OVER','SVMSMOTE':'OVER','KMeansSMOTE':'OVER'
                 ,'ADASYN':'OVER','RandomOverSampler':'OVER',
                 'SMOTEENN':'COMBINE','SMOTETomek':'COMBINE',
                 'CondensedNearestNeighbour':'UNDER','EditedNearestNeighbours':'UNDER',
                 'RepeatedEditedNearestNeighbours':'UNDER','AllKNN':'UNDER',
                 'InstanceHardnessThreshold':'UNDER','NearMiss':'UNDER',
                            'NeighbourhoodCleaningRule':'UNDER','OneSidedSelection':'UNDER','RandomUnderSampler':'UNDER',
                            'TomekLinks':'UNDER','ClusterCentroids':'UNDER'}        
        self.iid=0
    def getdata(self):
        ##If you read data directy from the original KEEL's zip file, use this code:
        zf = zipfile.ZipFile(self.file) 
        in_mem_fo = io.TextIOWrapper(io.BytesIO(zf.read(self.dataset)), encoding='utf-8')
        ####Convert KEEL format to arff ###
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
        #print(X.shape)
        return X,y


    def fscore(self,params_org):
        #print(params_org)
        parambk = copy.deepcopy(params_org)
        ifError =0
        #global best, HPOalg,params_best, errorcount,resampler_group,_idxi,_error, X,y
        self._idxi=int(self._idxi)+1
        p_random_state = int(params_org.pop('random_state'))
        params= params_org['classifier']
        classifier = params.pop('name')    
        p_sub_params= params_org.pop('resampler')
        p_sub_type = p_sub_params.pop('name')
        sampler = self.resampler_group[p_sub_type]
        if (classifier == 'SVM'):  
            gamma_value= params.pop('gamma_value')
            if(params['gamma'] == "value"):
                params['gamma'] = gamma_value
            #print(params)
            clf = SVC(max_iter = 10000, cache_size= 700, random_state = p_random_state,**params)
            #max_iter=10000 and cache_size= 700 https://github.com/EpistasisLab/pennai/issues/223
            #maxvalue https://github.com/hyperopt/hyperopt-sklearn/blob/fd718c44fc440bd6e2718ec1442b1af58cafcb18/hpsklearn/components.py#L262
        elif(classifier == 'RF'):        
            clf = RandomForestClassifier(random_state = p_random_state, **params)
        elif(classifier == 'KNN'):
            p_value = params.pop('p_value')
            params['n_neighbors']= params.pop('n_neighbors_knn')
            #print(self._idxi,'KNN',params)
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
            if 'criterion_dtc' in params:
                params['criterion']= params.pop('criterion_dtc')
                params['max_features']= params.pop('max_features_dtc')
                params['min_samples_split']= params.pop('min_samples_split_dtc')
                params['min_samples_leaf']= params.pop('min_samples_leaf_dtc')
            params['max_depth']= params.pop('max_depth_dtc')
            clf = DecisionTreeClassifier(random_state = p_random_state, **params)
        elif(classifier == 'LR'):
            if 'C_LR' in params:
                params['C']= params.pop('C_LR')
            if 'tol_lr' in params:
                params['tol']= params.pop('tol_lr')
            penalty_solver = params.pop('penalty_solver')
            params['penalty'] = penalty_solver.split("+")[0]
            params['solver'] = penalty_solver.split("+")[1]
            clf = LogisticRegression(random_state = p_random_state, **params)
        #resampling parameter
        #resGroup=AlgorithmChoice(["NO","OVER","COMBINE","UNDER"],"ResGroup")
        gmean = []
        notuse = ['x_no', 'x_ROS', 'x_COM1', 'x_COM2', 'x_TML', 'random_state_X']
        tobedel = [i for i in notuse if i in p_sub_params]
        for x in tobedel:
            xxxx=p_sub_params.pop(x)
        if p_sub_type not in ('EditedNearestNeighbours','RepeatedEditedNearestNeighbours','AllKNN',
                              'NearMiss','NeighbourhoodCleaningRule','TomekLinks'):
            p_sub_params['random_state']=p_random_state
        kind_sel_sets=['kind_sel1','kind_sel2','kind_sel3']
        kind_sel_value=[i for i in kind_sel_sets if i in p_sub_params]
        if len(kind_sel_value):
            p_sub_params['kind_sel']= p_sub_params.pop(kind_sel_value[0])

        k_neighbors=['k_neighbors_SMOTE','k_neighbors_Borderline','k_neighbors_SMOTENC','k_neighbors_SVMSMOTE','k_neighbors_KMeansSMOTE']
        k_value=[i for i in k_neighbors if i in p_sub_params]
        if len(k_value)>0:        
            p_sub_params['k_neighbors']= p_sub_params.pop(k_value[0])
        m_neighbors=['m_neighbors_Borderline','m_neighbors_SVMSMOTE']
        m_value=[i for i in m_neighbors if i in p_sub_params]
        if len(m_value)>0:
            p_sub_params['m_neighbors']= int(p_sub_params.pop(m_value[0]))

        n_neighbors=['n_neighbors_CNN','n_neighbors_UNDER1','n_neighbors_UNDER2','n_neighbors_UNDER3',
                     'n_neighbors_UNDER4','n_neighbors_UNDER5','n_neighbors_UNDER6','n_neighbors_OVER','x_neighbors_OVER']
        n_value=[i for i in n_neighbors if i in p_sub_params]
        if len(n_value)>0:
            p_sub_params['n_neighbors']= int(p_sub_params.pop(n_value[0]))

        if 'estimator_IHT' in p_sub_params:
            p_sub_params['estimator']= p_sub_params.pop('estimator_IHT')

        if (p_sub_type == 'SMOTE'):
            smo = SMOTE(**p_sub_params)
        elif (p_sub_type == 'ADASYN'):
            #print(p_sub_type,p_sub_params)
            if 'k_neighbors' in p_sub_params:
                p_sub_params['n_neighbors']=int(p_sub_params.pop('k_neighbors'))
            smo = ADASYN(**p_sub_params)
        elif (p_sub_type == 'BorderlineSMOTE'):
            smo = BorderlineSMOTE(**p_sub_params)
        elif (p_sub_type == 'SVMSMOTE'):
            smo = SVMSMOTE(**p_sub_params)
        elif (p_sub_type == 'SMOTENC'):
            p_sub_params['categorical_features']=True
            smo = SMOTENC(**p_sub_params)
        elif (p_sub_type == 'KMeansSMOTE'):
            #print(p_sub_params)
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
            #p_sub_params['n_neighbors']= int(p_sub_params.pop('n_neighbors_CNN'))
            if 'n_seeds_S_CNN' in p_sub_params:
                p_sub_params['n_seeds_S']= p_sub_params.pop('n_seeds_S_CNN')        
            smo = CondensedNearestNeighbour(**p_sub_params)
        elif (p_sub_type == 'EditedNearestNeighbours'):
            smo = EditedNearestNeighbours(**p_sub_params)
        elif (p_sub_type == 'RepeatedEditedNearestNeighbours'):
            smo = RepeatedEditedNearestNeighbours(**p_sub_params) 
        elif (p_sub_type == 'AllKNN'):
            #print(self._idxi,'ALLKNN',p_sub_params)
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
        cv = StratifiedKFold(n_splits=5)
        status=STATUS_OK
        X=self.X
        y=self.y
        try:        
            for train, test in cv.split(X,y):
                if(p_sub_type=='NO'):
                    X_smo_train, y_smo_train = X[train], y[train]
                else:
                    X_smo_train, y_smo_train = smo.fit_resample(X[train], y[train])
                y_test_pred = clf.fit(X_smo_train, y_smo_train).predict(X[test])
                gm = geometric_mean_score(y[test], y_test_pred, average='binary')
                gmean.append(gm)
            mean_g=np.mean(gmean)
        except Exception as eec:
            e=eec
            mean_g = 0
            ifError =1 
            Exsupport.errorcount +=1
        gm_loss = 1 - mean_g
        abc=time.time()-starttime
        if mean_g > self.best:
            self.best = mean_g
            self.params_best = copy.deepcopy(parambk)

        return {'loss': gm_loss,
                'mean': mean_g,
                'status': status,         
                # -- store other results like this
                'run_time': abc,
                'iter': self._idxi,
                'current_best': self.best,
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


def get_sp(randomstate,n_init_sample, isMax=True):
    search_space = ConfigSpace()
    con = ConditionalSpace("test")
    random_state=CategoricalParam(randomstate,'random_state')
    alg_namestr=AlgorithmChoice(['SVM','RF','KNN','DTC','LR'], 'classifier', default='SVM')    
    search_space.add_multiparameter([random_state,alg_namestr])    
    #SVM
    probability=CategoricalParam([True, False],'probability')
    C=FloatParam([0.03125 , 200],'C')
    kernel=CategoricalParam(['linear','rbf','poly', 'sigmoid'], 'kernel', default='linear')
    degree=IntegerParam([2,5],'degree')
    gamma=CategoricalParam([['auto','scale'],'value'], 'gamma', default='auto')
    gamma_value=FloatParam([3.1E-05, 8], 'gamma_value')
    coef0=FloatParam([-1,1], 'coef0')
    shrinking=CategoricalParam([True, False],'shrinking')
    tol_svm=FloatParam([1e-05, 1e-01], 'tol')
    search_space.add_multiparameter([probability,C,kernel,degree,gamma,gamma_value,coef0,shrinking,tol_svm])
    con.addMutilConditional([probability,C,kernel,degree,gamma,gamma_value,coef0,shrinking,tol_svm],alg_namestr,'SVM')
    #con.addConditional(gamma_value, gamma,'value')    
    ##RF
    n_estimators=IntegerParam([1,150],'n_estimators')
    criterion=CategoricalParam(['gini', 'entropy'],'criterion')
    max_features_RF=CategoricalParam([1, 'sqrt','log2',None],'max_features')  
    min_samples_split=IntegerParam([2, 20],'min_samples_split')
    min_samples_leaf=IntegerParam([1, 20],'min_samples_leaf')
    bootstrap=CategoricalParam([True, False],'bootstrap')
    class_weight=CategoricalParam([['balanced','balanced_subsample'],None],'class_weight')
    search_space.add_multiparameter([n_estimators,criterion,max_features_RF,min_samples_split,min_samples_leaf,bootstrap,class_weight])
    con.addMutilConditional([n_estimators,criterion,max_features_RF,min_samples_split,
                             min_samples_leaf,bootstrap,class_weight],alg_namestr,'RF')
    ###KNN
    n_neighbors_knn=IntegerParam([1,51],'n_neighbors_knn')
    weights=CategoricalParam(['uniform', 'distance'],'weights')
    algorithm=CategoricalParam(['auto', 'ball_tree', 'kd_tree', 'brute'],'algorithm')
    p=IntegerParam([0,20],'p_value')
    search_space.add_multiparameter([n_neighbors_knn,weights,algorithm,p])
    con.addMutilConditional([n_neighbors_knn,weights,algorithm,p],alg_namestr,'KNN')
    ####DTC    
    criterion_dtc=CategoricalParam(['gini', 'entropy'],'criterion_dtc')
    max_features_dtc=CategoricalParam([1, 'sqrt','log2',None],'max_features_dtc')
    max_depth=IntegerParam([2,20],'max_depth_dtc')
    min_samples_split_dtc=IntegerParam([2, 20],'min_samples_split_dtc')
    min_samples_leaf_dtc=IntegerParam([1, 20],'min_samples_leaf_dtc')
    #search_space.add_multiparameter([max_depth])
    #con.addMutilConditional([criterion,max_features_RF,min_samples_split,min_samples_leaf,max_depth],alg_namestr,"DTC")
    search_space.add_multiparameter([criterion_dtc,max_features_dtc,max_depth,min_samples_split_dtc,min_samples_leaf_dtc])
    con.addMutilConditional([criterion_dtc,max_features_dtc,max_depth,min_samples_split_dtc,min_samples_leaf_dtc],alg_namestr,"DTC")
    #####LR
    C_lr=FloatParam([0.03125 , 100],'C_LR')
    penalty_solver=CategoricalParam([['l1+liblinear','l2+liblinear'],
                                     ['l1+saga','l2+saga','elasticnet+saga','none+saga'],['l2+sag','none+sag'],
                                     ['l2+newton-cg','none+newton-cg'],['l2+lbfgs','none+lbfgs']],'penalty_solver')
    tol_lr=FloatParam([1e-05, 1e-01], 'tol_lr')
    l1_ratio=FloatParam([1e-09, 1], 'l1_ratio')
    search_space.add_multiparameter([C_lr,penalty_solver,tol_lr,l1_ratio])
    con.addMutilConditional([C_lr,penalty_solver,tol_lr,l1_ratio],alg_namestr,'LR')
    smo_type=AlgorithmChoice([['NO'],['SMOTE','BorderlineSMOTE','SMOTENC','SVMSMOTE','KMeansSMOTE','ADASYN','RandomOverSampler']
                              ,['SMOTEENN','SMOTETomek'],
                              ['CondensedNearestNeighbour','EditedNearestNeighbours','RepeatedEditedNearestNeighbours','AllKNN',
                              'InstanceHardnessThreshold','NearMiss','NeighbourhoodCleaningRule',
                               'OneSidedSelection','RandomUnderSampler','TomekLinks','ClusterCentroids']],'resampler')
    if n_init_sample==50 and isMax==True:
        smo_type=AlgorithmChoice([['NO'],['SMOTEENN','SMOTETomek'],['SMOTE','BorderlineSMOTE'],['SMOTENC','SVMSMOTE'],
                              ['RandomOverSampler','KMeansSMOTE','ADASYN'],
                              ['NearMiss','InstanceHardnessThreshold'],['TomekLinks','ClusterCentroids'],
                              ['EditedNearestNeighbours','RepeatedEditedNearestNeighbours'],['NeighbourhoodCleaningRule','AllKNN'],
                              ['OneSidedSelection','CondensedNearestNeighbour','RandomUnderSampler']],'resampler')
    elif n_init_sample==100 and isMax==True:
        smo_type=AlgorithmChoice(['SMOTE','BorderlineSMOTE','SMOTENC','SVMSMOTE','KMeansSMOTE','ADASYN'
                              ,'NO','SMOTEENN','SMOTETomek','NearMiss','InstanceHardnessThreshold','TomekLinks','ClusterCentroids',
                              'EditedNearestNeighbours','RepeatedEditedNearestNeighbours','NeighbourhoodCleaningRule',
                               'AllKNN','OneSidedSelection','CondensedNearestNeighbour',
                                  ['RandomUnderSampler','RandomOverSampler']],'resampler')
    else:
        pass
    search_space._add_singleparameter(smo_type)
    k_neighbors_SMOTE=IntegerParam([1,10],'k_neighbors_SMOTE')
    k_neighbors_Borderline=IntegerParam([1,10],'k_neighbors_Borderline')
    m_neighbors_Borderline=IntegerParam([1,10],'m_neighbors_Borderline')
    kind=CategoricalParam(['borderline-1', 'borderline-2'],'kind')
    categorical_features=CategoricalParam([True],'categorical_features')
    k_neighbors_SMOTENC=IntegerParam([1,10],'k_neighbors_SMOTENC')
    k_neighbors_SVMSMOTE=IntegerParam([1,10],'k_neighbors_SVMSMOTE')
    m_neighbors_SVMSMOTE=IntegerParam([1,10],'m_neighbors_SVMSMOTE') 
    out_step=FloatParam([0 , 1],'out_step')   
    k_neighbors_KMeansSMOTE=IntegerParam([1,10],'k_neighbors_KMeansSMOTE')  
    cluster_balance_threshold=FloatParam([1e-2, 1],'cluster_balance_threshold')
    n_neighbors_OVER=IntegerParam([1,10],'n_neighbors_OVER')
    search_space.add_multiparameter([k_neighbors_SMOTE,k_neighbors_Borderline,m_neighbors_Borderline,kind,
                                     categorical_features,k_neighbors_SMOTENC,
                                     k_neighbors_SVMSMOTE,m_neighbors_SVMSMOTE,out_step,
                                     k_neighbors_KMeansSMOTE,cluster_balance_threshold,n_neighbors_OVER])
    con.addConditional(k_neighbors_SMOTE, smo_type,'SMOTE')
    con.addMutilConditional([k_neighbors_Borderline,m_neighbors_Borderline,kind], smo_type,'BorderlineSMOTE')
    con.addMutilConditional([categorical_features,k_neighbors_SMOTENC,], smo_type,'SMOTENC')
    con.addMutilConditional([k_neighbors_SVMSMOTE,m_neighbors_SVMSMOTE,out_step], smo_type,'SVMSMOTE')
    con.addMutilConditional([k_neighbors_KMeansSMOTE,cluster_balance_threshold], smo_type,'KMeansSMOTE')
    con.addConditional(n_neighbors_OVER, smo_type,'ADASYN')
    n_neighbors_UNDER50=IntegerParam([1,50],'n_neighbors_CNN')
    n_seeds_S=IntegerParam([1,50],'n_seeds_S_CNN')
    n_neighbors_UNDER1=IntegerParam([1,20],'n_neighbors_UNDER1')
    kind_sel1=CategoricalParam(['all','mode'],'kind_sel1')
    n_neighbors_UNDER2=IntegerParam([1,20],'n_neighbors_UNDER2')
    kind_sel2=CategoricalParam(['all','mode'],'kind_sel2')
    n_neighbors_UNDER3=IntegerParam([1,20],'n_neighbors_UNDER3')
    kind_sel3=CategoricalParam(['all','mode'],'kind_sel3')
    allow_minority=CategoricalParam([True, False],'allow_minority')
    estimator_IHT=CategoricalParam(['knn', 'decision-tree', 'adaboost','gradient-boosting','linear-svm',None],'estimator_IHT')
    cv_under=IntegerParam([2,20],'cv')
    version=CategoricalParam([1,2,3],'version')
    n_neighbors_UNDER4=IntegerParam([1,20],'n_neighbors_UNDER4')
    n_neighbors_ver3=IntegerParam([1,20],'n_neighbors_ver3')
    n_neighbors_UNDER5=IntegerParam([1,20],'n_neighbors_UNDER5')
    threshold_cleaning_NCR=FloatParam([0 , 1],'threshold_cleaning')
    n_neighbors_UNDER6=IntegerParam([1,20],'n_neighbors_UNDER6')
    n_seeds_S_under=IntegerParam([1,20],'n_seeds_S')
    replacement=CategoricalParam([True, False],'replacement')
    estimator_CL=CategoricalParam(['KMeans', 'MiniBatchKMeans'],'estimator')
    voting_CL=CategoricalParam(['hard', 'soft'],'voting')
    search_space.add_multiparameter([n_neighbors_UNDER50,n_seeds_S,n_neighbors_UNDER1,kind_sel1,
                                     n_neighbors_UNDER2,kind_sel2,n_neighbors_UNDER3,kind_sel3,                                     
                                     allow_minority,estimator_IHT,cv_under,version,n_neighbors_UNDER4,n_neighbors_ver3,
                                     n_neighbors_UNDER5,threshold_cleaning_NCR,n_neighbors_UNDER6,n_seeds_S_under,
                                     replacement,estimator_CL,voting_CL
                                    ])
    con.addMutilConditional([n_neighbors_UNDER50, n_seeds_S], smo_type,'CondensedNearestNeighbour')
    con.addMutilConditional([n_neighbors_UNDER1, kind_sel1], smo_type,'EditedNearestNeighbours')
    con.addMutilConditional([n_neighbors_UNDER2, kind_sel2], smo_type,'RepeatedEditedNearestNeighbours')
    con.addMutilConditional([n_neighbors_UNDER3, kind_sel3, allow_minority], smo_type,'AllKNN')
    con.addMutilConditional([estimator_IHT, cv_under], smo_type,'InstanceHardnessThreshold')
    con.addMutilConditional([version,n_neighbors_UNDER4,n_neighbors_ver3], smo_type,'NearMiss')
    con.addMutilConditional([n_neighbors_UNDER5,threshold_cleaning_NCR], smo_type,'NeighbourhoodCleaningRule')
    con.addMutilConditional([n_neighbors_UNDER6,n_seeds_S_under], smo_type,'OneSidedSelection')
    con.addConditional(replacement, smo_type,'RandomUnderSampler')
    con.addMutilConditional([estimator_CL,voting_CL], smo_type,'ClusterCentroids')
    return search_space, con

seeds= [seed] if seed>0 else [*range(1,20)] 
n_EI_candidates=24
eta=2
isMax=True
isFair=True
for n_init_sample in [50,100]:  
    for randomstate in seeds:  
        print('\033[91m',HPOalg,'==Random Seed:',randomstate,'=== START DATASET: ', dataset, '=======', '\033[0m')
        sample_sp=1 if n_init_sample==20 else None
        print('0.25 n_init_sample: ',n_init_sample,' - n_EI_candidates',n_EI_candidates)
        search_space,con=get_sp(randomstate,n_init_sample,isMax=isMax)
        trials = Trials()
        exp=Exsupport(file,dataset,randomstate,n_init_sample)
        opt = BO4ML(search_space, exp.fscore,conditional=con,hpo_prefix='name',isFair=isFair,
                    SearchType='full', hpo_algo='tpe',random_seed=randomstate,
                    max_eval=500, verbose=True,HPOopitmizer='hpo', n_init_sample=n_init_sample,
                    sample_sp=sample_sp,max_threads=1,ifAllSolution=True, n_EI_candidates=n_EI_candidates)
        starttime = time.time()
        xopt, fopt, _, eval_count = opt.run()
        runtime=time.time()-starttime
        print(randomstate,xopt, 1-fopt, _, eval_count,runtime,exp.errorcount)
        try:
            ran_results = pd.DataFrame({'current_best': [x['current_best'] for x in opt.results.values()],
                                        'run_time':[x['run_time'] for x in opt.results.values()],
                                        'classifier': [x['classifier'] for x in opt.results.values()],
                                        'SamplingGrp': [x['SamplingGrp'] for x in opt.results.values()],
                                        'SamplingType': [x['SamplingType'] for x in opt.results.values()], 
                                        'ifError': [x['ifError'] for x in opt.results.values()], 
                                        'Error': [x['Error'] for x in opt.results.values()], 
                                        'loss': [x['loss'] for x in opt.results.values()], 
                                        'mean': [x['mean'] for x in opt.results.values()], 
                                        'iteration': [i for i in opt.results.keys()],
                                        'params':[x['params'] for x in opt.results.values()]})
            ran_results.to_csv(DataFolder+'/OURapproach/'+str(n_init_sample)+'_ourMax_'+HPOalg+'_'+dataset+'_'+str(randomstate)+'.csv', 
                               index = True, header=True)
        except:
            print('ERROR: No logfile')
        finallog= HomeFolder+"/OURapproach_isMax.csv"
        if (os.path.exists(finallog)==False):
            with open(finallog, "a") as f:    
                wr = csv.writer(f, dialect='excel')
                wr.writerow(['dataname','HPOalg','random_state','initsample','EI','eta','mean',
                             'params','runtime','errorcount','isMax','isFair'])
        with open(finallog, "a") as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow([dataset,HPOalg,randomstate,n_init_sample,n_EI_candidates,eta,1-fopt,xopt,runtime,exp.errorcount,isMax, isFair])




