from collections import OrderedDict
from math import ceil, floor
import copy, math
import numpy as np
from Component.BayesOpt import bayesopt
from BanditOpt.ConditionalSpace import ConditionalSpace
from BanditOpt.ConfigSpace import ConfigSpace
import Component.BayesOpt.bayesopt as MIP
import Component.mHyperopt.hyperopt as HO
from Component.BayesOpt import RandomForest
from Component.BayesOpt import ContinuousSpace,OrdinalSpace,NominalSpace,SearchSpace
from BanditOpt.Forbidden import Forbidden
from BanditOpt.HyperoptConverter import SubToHyperopt, OrginalToHyperopt
from Component.mHyperopt import tpe, Trials
class BO4ML(object):
    def __init__(self, search_space: ConfigSpace,
                 obj_func,#surrogate=None,
                 conditional: ConditionalSpace = None,
                 forbidden:Forbidden = None,
                 eta=3,
                 SearchType="Bandit",
                 HPOopitmizer= "BayesOpt",
                 sp_cluster=0,
                 parallel_obj_fun=None,
                 eq_fun=None,
                 ineq_fun=None,
                 model= None,
                 eval_type='dict',
                 DoE_size=None,
                 warm_data=(),
                 n_point=1,
                 acquisition_fun='EI',
                 acquisition_par={},
                 acquisition_optimization={},
                 ftarget=None,
                 #max_FEs=None,
                 minimize=True,
                 max_eval=None,
                 #max_iter=None,
                 #init_points=None,
                 #infill='EI',
                 #noisy=False,
                 #t0=2,
                 #tf=1e-1,
                 #schedule='exp',
                 n_init_sample=5,
                 n_job=1,
                 #n_restart=None,
                 #max_infill_eval=None,
                 #wait_iter=3,
                 #optimizer='MIES',
                 data_file=None,
                 verbose=False,
                 random_seed=None,
                 logger=None,
                 hpo_max_eval=None,
                 hpo_prefix='value',
                 hpo_algo= tpe.suggest,
                 hpo_trials= None,
                 hpo_pass_expr_memo_ctrl = None,
                 hpo_verbose = 0,
                 hpo_max_queue_len =1,
                 hpo_show_progressbar=True,
                 hpo_return_argmin=True
                 ):
        #Hyperband: parameter
        self.eta=eta
        self.iter_count = 0
        self.eval_count = 0
        self.stop_dict = {}
        self.HPOopitmizer= HPOopitmizer
        self.max_eval = max_eval
        self.n_init_sample=n_init_sample
        self.sp_cluster=sp_cluster
        self.isminize=minimize
        #MIP-EGO:parameter
        if (len(conditional.conditional)<1):
            conditional=None
        #self.orgSearchSpace=search_space
        #self.orgConditional=conditional
        #self.orgForbidden = forbidden
        isBandit=True
        if (conditional == None or SearchType !="Bandit"):
            isBandit=False
        self.isBandit=isBandit
        self.searchspace = search_space.Combine(conditional, forbidden, isBandit,sp_cluster)

        """if (conditional==None):
            self.searchspace = search_space
        else:
            self.searchspace = search_space.combinewithconditional(conditional, forbidden)
            """
        self.gb_N_value_count = sum([len(search_space._OrgLevels[i]) for i in search_space._OrgLevels])
        self._lsCurrentBest = OrderedDict()
        self._lsOrderdBest = OrderedDict()
        self._lsincumbent = OrderedDict()
        self.opt = OrderedDict()
        self.BO4ML_kwargs = OrderedDict()
        self.isHyperopt=False
        ###mHyperopt:
        if HPOopitmizer in ['hyperopt','Hyperopt','hpo','HyperOpt']:
            self.isHyperopt=True
            self.HPO = {}
            self.HPO['obj_func']=obj_func
            if(isBandit):
                HPOsearchspace = SubToHyperopt(self.searchspace, conditional,hpo_prefix)
            else:
                HPOsearchspace = OrginalToHyperopt(search_space, conditional,hpo_prefix)
            self.searchspace=HPOsearchspace
            self.HPO['org_search_space'] = HPOsearchspace
            self.HPO['algo'] = hpo_algo
            self.HPO['max_evals'] = max_eval
            self.HPO['trials']= hpo_trials
            self.HPO['rstate'] = random_seed
            #self.HPO['n_init_sample'] = n_init_sample
            self.HPO['pass_expr_memo_ctrl']=hpo_pass_expr_memo_ctrl
            self.HPO['verbose']=hpo_verbose
            self.HPO['return_argmin']=hpo_return_argmin
            self.HPO['max_queue_len']=hpo_max_queue_len
            self.HPO['show_progressbar']=hpo_show_progressbar
        else:
            self.MIP = {}
            if n_init_sample!=0:
                DoE_size=n_init_sample
            self.MIP['isBandit'] = isBandit
            self.MIP['obj_fun'] = obj_func
            self.MIP['model']=model
            self.MIP['DoE_size'] = DoE_size
            self.MIP['acquisition_fun'] = acquisition_fun
            self.MIP['acquisition_par'] = acquisition_par
            self.MIP['acquisition_optimization'] = acquisition_optimization
            self.MIP['max_FEs'] = max_eval
            #self.MIP['surrogate'] = surrogate
            self.MIP['parallel_obj_fun'] = parallel_obj_fun
            self.MIP['ftarget'] = ftarget
            self.MIP['eq_fun'] = eq_fun
            self.MIP['ineq_fun'] = ineq_fun
            self.MIP['minimize'] = minimize
            #self.MIP['max_eval'] = max_eval
            #self.MIP['max_iter'] = max_iter
            #self.MIP['init_points'] = init_points
            self.MIP['warm_data'] = warm_data
            #self.MIP['infill'] = infill
            #self.MIP['noisy'] = noisy
            #self.MIP['t0'] = t0
            #self.MIP['tf'] = tf
            #self.MIP['schedule'] = schedule
            self.MIP['eval_type'] = eval_type
            #self.MIP['n_init_sample'] = n_init_sample
            self.MIP['n_point'] = n_point
            self.MIP['n_job'] = n_job
            #self.MIP['n_restart'] = n_restart
            #self.MIP['max_infill_eval'] = max_infill_eval
            #self.MIP['wait_iter'] = wait_iter
            #self.MIP['optimizer'] = optimizer
            self.MIP['data_file'] = data_file
            self.MIP['verbose'] = verbose
            self.MIP['random_seed'] = random_seed
            self.MIP['logger'] = logger
            self.MIP['forbidden'] = forbidden
            self.MIP['conditional'] = conditional
            self.MIP['hyperparameters'] = search_space

    def run(self):
        '''if(self.HPOopitmizer in ['hyperopt','Hyperopt','hpo','HyperOpt']):
            if (self.isBandit==False):
                print('NOT implement yet!, please use hyperopt instead')
                #return self.runHO()
        else:'''
        if (self.isBandit == True):
            if ((self.sp_cluster > 0) and (self.sp_cluster ** 3 > self.max_eval)):
                # if (len(self.searchspace) * self.MIP['n_point'] * self.MIP['n_init_sample'] > self.max_eval):
                print("NOT ENOUGH BUDGET for: " + str(len(self.searchspace)) + "search spaces")
                return None, None, None, None
            else:
                return self.runBO4ML()
        else:
            return self.runBO(self.searchspace)

    def runBO(self, search_space):
        if (self.isHyperopt):
            trials = Trials()
            kwargs = copy.deepcopy(self.HPO)
            kwargs['sp_id'] = 0
            kwargs['search_space'] = search_space
            kwargs['max_evals'] = self.n_init_sample
            kwargs['trials'] = trials
            BO = HO.HyperOpt(**kwargs)
        else:
            self.MIP['sp_id'] = 0
            self.MIP['search_space'] = search_space
            kwargs = self.MIP
            BO = MIP.BayesOpt(**kwargs)
        return BO.run()
    def runBO4ML(self):

        for sp in self.searchspace:
            sp_id = self.searchspace.index(sp)
            if 'kwargs' in locals():
                kwargs.clear()
            if (self.isHyperopt):
                trials = Trials()
                kwargs = copy.deepcopy(self.HPO)
                kwargs['sp_id'] = sp_id
                kwargs['search_space'] = sp
                kwargs['max_evals'] = self.n_init_sample
                kwargs['trials'] = trials

                self.BO4ML_kwargs[sp_id] = copy.deepcopy(kwargs)
                self.opt[sp_id] = HO.HyperOpt(**kwargs)
            else:
                kwargs=copy.deepcopy(self.MIP)
                kwargs['sp_id'] = sp_id
                kwargs['search_space'] = sp
                kwargs['max_eval'] = self.n_init_sample
                '''if (kwargs['n_point']>1):
                    kwargs['max_eval']= kwargs['n_init_sample'] +1
                else:
                    kwargs['max_eval']= kwargs['n_init_sample']'''
                self.BO4ML_kwargs[sp_id] = copy.deepcopy(kwargs)
                self.opt[sp_id] = MIP.BayesOpt(**kwargs)

            #funcType=type(BO.BayesOpt.pre_eval_check)
            #self.opt[sp_id].pre_eval_check = funcType(ext.check_configuration,self.opt[sp_id],BO.BayesOpt)
            try:
                xopt, fopt, stop_dict, ieval_count = self.opt[sp_id].run()
                self._lsincumbent[sp_id] = xopt
                self._lsCurrentBest[sp_id] =fopt #min(self.opt[sp_id].eval_hist)
                self.eval_count += ieval_count
                print('INIT message:', str(sp_id), '--best: ', str(fopt))
            except Exception as e:
                #ieval_count =self.opt[sp_id].eval_count
                print('INIT Round ERROR:', str(sp_id), '--msg:', e)
             #self.opt[sp_id].eval_count
            # lsRunning[sp_id] = [incumbent,model,stop_dict, opt[sp_id]]
        max_eval = self.max_eval - self.eval_count
        lsRace = self.calculateSH()
        num_races = len(lsRace)
        eval_race = max_eval / num_races
        errList = []
        for iround, num_candidate in lsRace.items():
            cd_add_eval = int(floor(eval_race / num_candidate))
            print("Round: ", iround + 1, ", Candidates:", num_candidate, "Func Eval/candidate:",cd_add_eval)
            if (self.isminize == True):
                lsThisRound = list(OrderedDict(sorted(self._lsCurrentBest.items(), key=lambda item: item[1])).items())[
                              :num_candidate]
            else:
                lsThisRound = list(OrderedDict(sorted(self._lsCurrentBest.items(), key=lambda item: item[1],
                                                      reverse=True)).items())[:num_candidate]
            for cdid, bestloss in lsThisRound:
                #cd_add_eval = int(floor(eval_race / num_candidate))
                if (num_candidate<=1):
                    remain_eval = self.max_eval - self.eval_count
                    cd_add_eval = max(cd_add_eval,remain_eval)
                print("previous best loss was:", bestloss, "of", cdid)
                if(self.isHyperopt):
                    cd_ran_eval = len(self.opt[cdid].trials)
                else:
                    cd_ran_eval = self.opt[cdid].BO.eval_count
                try:
                    xopt, fopt, stop_dict,ieval_count = self.opt[cdid].AddBudget_run(cd_add_eval,iround)
                    self._lsincumbent[cdid] = xopt
                except Exception as e:
                    self.opt[cdid].max_eval = self.opt[cdid].eval_count
                    cd_add_eval = self.opt[cdid].eval_count - cd_ran_eval
                    fopt = self.opt[cdid].eval_hist[self.opt[cdid].incumbent_id]
                    if (cdid not in errList):
                        errList.append(cdid)
                    print('BO4ML-ERROR at round:',str(iround),"-Candidate ID:",cdid,"--msg:", e)
                #update infor
                self.eval_count += cd_add_eval
                self._lsCurrentBest[cdid] = fopt
        errIDs = errList
        while (len(errList) > 0):
            if (len(errIDs) >= len(self.searchspace)):
                break
            print("Additional round, Runing: 1 Candidate")
            shortLst = OrderedDict([f for f in self._lsCurrentBest.items() if f[0] not in errIDs])
            errList = []
            if (self.isminize  == True):
                lsThisRound = list(OrderedDict(sorted(shortLst.items(), key=lambda item: item[1])).items())[:1]
            else:
                lsThisRound = list(OrderedDict(sorted(shortLst.items(), key=lambda item: item[1],
                                                      reverse=True)).items())[:1]
            for cdid, bestloss in lsThisRound:
                num_races +=1
                remain_eval = self.max_eval - self.eval_count
                remain_iter = self.max_iter - self.iter_count
                cd_add_eval = remain_eval
                cd_add_iter = remain_iter
                # lc_N_value_count= [len(sp.le)]
                print("previous best loss was:", bestloss, "of CandidateID", cdid)
                # cdvalue
                cd_ran_iter = self.opt[cdid].iter_count
                cd_ran_eval = self.opt[cdid].eval_count
                try:
                    xopt, fopt, stop_dict, ieval_count = self.opt[cdid].AddBudget_run(cd_add_eval,num_races)
                    self._lsincumbent[cdid] = xopt
                except Exception as e:
                    self._lsincumbent[cdid] = None
                    self.opt[cdid].max_iter = self.opt[cdid].iter_count
                    self.opt[cdid].max_eval = self.opt[cdid].eval_count
                    fopt = self.opt[cdid].eval_hist[self.opt[cdid].incumbent_id]
                    cd_add_iter = self.opt[cdid].iter_count - cd_ran_iter
                    cd_add_eval = self.opt[cdid].eval_count - cd_ran_eval
                    if (cdid not in errIDs):
                        errIDs.append(cdid)
                    errList.append(cdid)
                    print('BO4ML-ERROR: ==Additional Round==', e)
                # update infor
                self.iter_count += cd_add_iter
                self.eval_count += cd_add_eval
                self._lsCurrentBest[cdid] = fopt
        # conclusion
        if (self.isminize  == True):
            lsThisRound = list(OrderedDict(sorted(self._lsCurrentBest.items(), key=lambda item: item[1])).items())[:1]
        else:
            lsThisRound = list(OrderedDict(sorted(self._lsCurrentBest.items(), key=lambda item: item[1],
                                                  reverse=True)).items())[:1]
        best_cdid = lsThisRound[0][0]
        best_incumbent, best_value = self._lsincumbent[best_cdid], self._lsCurrentBest[best_cdid]
        lstrials=''
        if(self.isHyperopt):
            lstrials = [x.trials for _, x in self.opt.items()]
        return best_incumbent, best_value, lstrials, self.eval_count

    def mega_stop(self):
        if self.iter_count >= self.max_iter:
            self.stop_dict['max_iter'] = True
        if self.eval_count >= self.max_eval:
            self.stop_dict['max_eval'] = True
        return len(self.stop_dict)

    def calculateSH(self) -> OrderedDict():
        remain_candidate = len(self.searchspace)
        ratio = 1 / self.eta
        lsEval = OrderedDict()
        a = 0
        lsEval[a] = remain_candidate
        while remain_candidate > 1:
            a += 1
            remain_candidate = ceil(remain_candidate * ratio)
            lsEval[a] = remain_candidate
        return lsEval


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
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
    #opt = BO4ML(search_space, new_obj,forbidden=fobr,conditional=con,SearchType="Bandit", max_eval=50)
    suggest = tpe.suggest
    opt = BO4ML(search_space, new_obj, forbidden=fobr, conditional=con, SearchType="Bandit",
                HPOopitmizer='hyperopt', max_eval=30,hpo_algo=suggest)
    xopt, fopt, _, eval_count = opt.run()
    print(xopt,fopt)

