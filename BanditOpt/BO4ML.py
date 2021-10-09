from collections import OrderedDict, Counter
from math import ceil, floor
import copy, math
import numpy as np
import time
from BanditOpt.ConditionalSpace import ConditionalSpace
from BanditOpt.ConfigSpace import ConfigSpace
import Component.mHyperopt.hyperopt as HO
from joblib import Parallel, delayed
# from Component.BayesOpt import ContinuousSpace,OrdinalSpace,NominalSpace,SearchSpace
from BanditOpt.HyperParameter import HyperParameter, AlgorithmChoice, FloatParam, CategoricalParam, IntegerParam
from BanditOpt.Forbidden import Forbidden
from BanditOpt.HyperoptConverter import SubToHyperopt, OrginalToHyperopt, ForFullSampling
from Component.mHyperopt import rand,tpe, anneal,atpe, Trials, space_eval
from functools import partial


class BO4ML(object):
    def __init__(self, search_space: ConfigSpace,
                 obj_func,#surrogate=None,
                 conditional: ConditionalSpace = None,
                 forbidden:Forbidden = None,
                 eta=3,
                 SearchType="full",
                 HPOopitmizer= "Hyperopt",
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
                 min_sp=1,
                 n_init_sp=None,
                 timeout=None,
                 #max_iter=None,
                 #init_points=None,
                 #infill='EI',
                 #noisy=False,
                 #t0=2,
                 #tf=1e-1,
                 #schedule='exp',
                 n_init_sample=20,
                 n_job=1,
                 isFair=True,
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
                 hpo_return_argmin=True,
                 ifAllSolution = False,
                 sample_sp=None,
                 init_ratio=None,
                 max_threads="max",
                 n_EI_candidates="auto",
                 #timeout=None,
                 shuffle=None
                 ):
        #Hyperband: parameter
        self.eta=eta
        self.iter_count = 0
        self.eval_count = 0
        self.stop_dict = {}
        self.max_threads=max_threads
        self.HPOopitmizer= HPOopitmizer
        self.start_time=time.time()
        self.max_eval = max_eval
        self.n_init_sample=n_init_sample
        self.sp_cluster=sp_cluster
        self.isminize=minimize
        self.timeout=timeout
        self.ifAllSolution=ifAllSolution
        init_ratio=np.round((n_init_sample/max_eval),2) if init_ratio==None else init_ratio
        init_ratio = 1 if (ifAllSolution==True and hpo_algo =="rand") else init_ratio
        self.init_ratio=init_ratio
        self.sample_sp=sample_sp
        self.isFair=isFair
        self.shuffle=True if timeout !=None and shuffle==None else shuffle
        #MIP-EGO:parameter
        if (conditional==None):
            conditional=None
        #self.orgSearchSpace=search_space
        #self.orgConditional=conditional
        #self.orgForbidden = forbidden
        self._def_n_EI_candidates=24
        isBandit =False if conditional == None or SearchType !="Bandit" else True
        isFULLSearch=True if SearchType =="full" else False
        self.isBandit=isBandit
        self.isFullSearch=isFULLSearch
        self.orgSearchspace = search_space.Combine(conditional, forbidden, isBandit, sp_cluster,
                                                   ifAllSolution=ifAllSolution, random_seed= random_seed,min_sp=min_sp,
                                                   n_init_sp=n_init_sp, max_eval=max_eval, sample_sp=sample_sp, init_ratio=init_ratio)
        #_MaxCombination=np.product([len(x.allbounds) for x in search_space._hyperparameters.values() if isinstance(x,AlgorithmChoice)])
        n_EI_candidates = max(len(search_space._hyperparameter_idx),self._def_n_EI_candidates) if n_EI_candidates == "auto" else n_EI_candidates
        self.n_EI_candidates = n_EI_candidates if n_EI_candidates != None and n_EI_candidates > self._def_n_EI_candidates else self._def_n_EI_candidates
        self._lsCurrentBest = OrderedDict()
        self._lsOrderdBest = OrderedDict()
        self._lsincumbent = OrderedDict()
        self.opt = OrderedDict()
        self.BO4ML_kwargs = OrderedDict()
        self.isHyperopt=False
        self.seed=random_seed
        ###mHyperopt:
        if HPOopitmizer in ['hyperopt','Hyperopt','hpo','HyperOpt']:
            self.isHyperopt=True
            self.HPO = {}
            self.HPO['obj_func']=obj_func
            #self.HPO['timeout']=timeout
            if(isBandit):
                HPOsearchspace = SubToHyperopt(self.orgSearchspace, conditional,hpo_prefix)
            else:
                HPOsearchspace = OrginalToHyperopt(search_space._hyperparameters, conditional, hpo_prefix)
                if isFULLSearch:
                    _LstHPOsearchspace, _spRatio = ForFullSampling(search_space._hyperparameters, conditional, hpo_prefix,
                                                         ifAllSolution=ifAllSolution, random_seed= random_seed,min_sp=min_sp, init_sample=n_init_sample,
                                                         n_init_sp=n_init_sp, max_eval=max_eval, sample_sp=sample_sp, _defratio=init_ratio, _fair=isFair)
                    self._LstHPOsearchspace=_LstHPOsearchspace
                    self._spRatio=_spRatio
            self.searchspace=HPOsearchspace
            _lsthpo_algo={"rand":rand.suggest,"tpe":tpe.suggest,"atpe":atpe.suggest,"anneal":anneal.suggest}
            hpo_algo=hpo_algo.lower()
            self.hpo_suggest=hpo_algo

            if hpo_algo == "tpe":
                _hpo_algo = partial(tpe.suggest, n_startup_jobs=n_init_sample, n_EI_candidates=self.n_EI_candidates)
            else:
                _hpo_algo=_lsthpo_algo[hpo_algo]
            if random_seed!=None:
                rstate = np.random.RandomState(random_seed)
                self.HPO['rstate']=rstate
            self.HPO['org_search_space'] = HPOsearchspace
            self.HPO['algo'] = _hpo_algo
            self.HPO['algo_str'] = hpo_algo
            self.HPO['n_init_sample']=n_init_sample
            self.HPO['max_evals'] = max_eval
            self.trials = hpo_trials if hpo_trials != None else Trials()
            self.HPO['trials']= self.trials
            self.HPO['timeout']=timeout

            #self.HPO['rstate'] = random_seed
            #self.HPO['n_init_sample'] = n_init_sample
            self.HPO['pass_expr_memo_ctrl']=hpo_pass_expr_memo_ctrl
            self.HPO['verbose']=hpo_verbose
            self.HPO['return_argmin']=hpo_return_argmin
            self.hpo_max_queue_len=hpo_max_queue_len
            self.HPO['max_queue_len']=hpo_max_queue_len if hpo_max_queue_len>1 else max_threads
            self.HPO['show_progressbar']=hpo_show_progressbar
            self.HPO['eval_count']=0
        else:
            pass

    def run(self):
        '''if(self.HPOopitmizer in ['hyperopt','Hyperopt','hpo','HyperOpt']):
            if (self.isBandit==False):
                print('NOT implement yet!, please use hyperopt instead')
                #return self.runHO()
        else:'''
        np.random.seed(self.seed)
        return self.runBO(self.searchspace)

    @staticmethod
    def _save_results(trials:list):
        return dict(enumerate([x['result'] for x in trials]))
    @staticmethod
    def create_trials(orgtrials,trials:list):
        newtrials=orgtrials
        tid= max([trial['tid'] for trial in newtrials.trials]) if len(newtrials)>0 else -1
        for trial in trials:
            tid = tid + 1 if tid >=0 else 0
            hyperopt_trial = Trials().new_trial_docs(
                tids=[None],
                specs=[None],
                results=[None],
                miscs=[None])
            hyperopt_trial[0] = trial
            hyperopt_trial[0]['tid'] = tid
            hyperopt_trial[0]['misc']['tid'] = tid
            for key in hyperopt_trial[0]['misc']['idxs'].keys():
                oldVal = hyperopt_trial[0]['misc']['idxs'][key]
                if len(oldVal) > 0:
                    hyperopt_trial[0]['misc']['idxs'][key] = [tid]
            newtrials.insert_trial_docs(hyperopt_trial)
            newtrials.refresh()
        return newtrials
    @staticmethod
    def merge_trials(trials1, trials2):
        newtrials = trials1
        max_tid = max([trial['tid'] for trial in newtrials.trials]) if len(newtrials)>0 else -1
        tid = max_tid
        for trial in trials2:
            if 1==1:
            #if(trial['misc']['vals'] not in [x['misc']['vals'] for x in newtrials]):
                #tid = trial['tid'] + max_tid + 1
                tid=tid+1
                hyperopt_trial = Trials().new_trial_docs(
                        tids=[None],
                        specs=[None],
                        results=[None],
                        miscs=[None])
                hyperopt_trial[0] = trial
                hyperopt_trial[0]['tid'] = tid
                hyperopt_trial[0]['misc']['tid'] = tid
                for key in hyperopt_trial[0]['misc']['idxs'].keys():
                    oldVal=hyperopt_trial[0]['misc']['idxs'][key]
                    if len(oldVal) > 0:
                        hyperopt_trial[0]['misc']['idxs'][key] = [tid]
                newtrials.insert_trial_docs(hyperopt_trial)
                newtrials.refresh()
        return newtrials
    def runBO(self, search_space):
        if (self.isHyperopt):
            np.random.seed(self.seed)
            #self.trials = Trials()
            kwargs = copy.deepcopy(self.HPO)
            kwargs['isParallel'] = False
            _max_eval=0
            if self.isFullSearch:
                _totalSP = len(self._LstHPOsearchspace)
                sample_sp=self.n_init_sample/_totalSP if self.sample_sp== None else self.sample_sp
                _max_init =max(self.n_init_sample,sample_sp*_totalSP) if self.init_ratio<1 else self.max_eval
                #print("====INIT:", _max_init)
                if _max_init>self.max_eval and self.max_eval>0:
                    raise TypeError("Not Enough budget")
                _step_size=floor(_max_init/_totalSP) if self.sample_sp==None else sample_sp
                _max_eval,_imax_eval=0,0
                _eval_counted = 0
                self.max_threads=_totalSP if self.max_threads=="max" else self.max_threads
                assert isinstance(self.max_threads,int)
                if self.sample_sp==None:
                    #_lsstep_size = [max(1, round(x * _step_size)) for x in
                    #                self._spRatio]  # [round(x * _step_size) for x in self._spRatio]
                    #if sum(_lsstep_size)>_max_init:
                    _lsstep_size = [max(1, math.floor(x * _step_size)) for x in
                                        self._spRatio]
                else:
                    _lsstep_size = [_step_size for x in self._spRatio]
                if _max_init > sum(_lsstep_size):
                    _remainsamples = _max_init - sum(_lsstep_size)
                    _most_common=dict(sorted(Counter(dict(zip(range(_totalSP),self._spRatio))).most_common(round(_totalSP/self.eta if _totalSP>self.eta else _totalSP))))
                    _asum = sum(_most_common.values())
                    _most_common = {i: x / _asum for i, x in _most_common.items()}
                    _remainBG=dict(Counter(np.random.choice(list(_most_common.keys()), replace=True,
                                                            p=list(_most_common.values()),size=_remainsamples)))
                    for i,v in _remainBG.items():
                        _lsstep_size[i]=_lsstep_size[i]+v
                kwargs['timeout']=self.timeout-(time.time()-self.start_time) if self.timeout!= None else None
                kwargs['sp_id'] = 0
                kwargs['trials'] = self.trials
                while sum(_lsstep_size)>0:
                    _randomOrder = list(range(len(_lsstep_size)))
                    np.random.shuffle(_randomOrder)
                    # while _imax_eval<_max_init:
                    for iid in [x for x in _randomOrder if _lsstep_size[x]>0]:
                        if _lsstep_size[iid]<1:
                            continue
                    #for iid, _hposp in enumerate(self._LstHPOsearchspace):
                        _hposp=self._LstHPOsearchspace[iid]
                        print("===", iid, "===", _lsstep_size[iid])
                        _thisStepsize = 1 if self.shuffle==True else _lsstep_size[iid]
                        #print(_thisStepsize)
                        _lsstep_size[iid] = _lsstep_size[iid]-_thisStepsize
                        _eval_counted += _thisStepsize
                        _max_eval = _max_eval + _thisStepsize
                        _imax_eval = _imax_eval + _thisStepsize
                        # rstate = np.random.RandomState(self.seed)
                        # kwargs['rstate'] = rstate
                        kwargs['max_queue_len']=min(_thisStepsize,self.hpo_max_queue_len)
                        kwargs['search_space'] = _hposp
                        kwargs['max_evals'] = _max_eval
                        kwargs['fix_max_evals'] = _imax_eval
                        kwargs['n_init_sample'] = _imax_eval
                        kwargs['timeout'] = self.timeout - (
                                time.time() - self.start_time) if self.timeout != None else None
                        if _imax_eval > _max_init or (kwargs['timeout'] != None and kwargs['timeout'] <= 0):
                            # print("BREAK")
                            break
                        if self.hpo_suggest == "tpe":
                            _hpo_algo = partial(tpe.suggest, n_startup_jobs=_max_eval,
                                                n_EI_candidates=self.n_EI_candidates)
                            kwargs['algo'] = _hpo_algo
                        # kwargs['algo'] = rand.suggest
                        BO = HO.HyperOpt(**kwargs)
                        # print("====MAX:", _max_eval)
                        _, _, _max_eval, _imax_eval = BO.run()
                if _imax_eval < _max_init:
                    # rstate = np.random.RandomState(self.seed)
                    # kwargs['rstate'] = rstate
                    _thisStepsize = _max_init - _eval_counted
                    '''_remainBG= dict(Counter(np.random.choice(range(_totalSP),replace=True,p=self._spRatio/_totalSP, size=_thisStepsize))
                                    
                    _remainbg=[None] * _thisStepsize
                    for i in range(_thisStepsize):
                        _remainbg[i]=np.random.choice(self._LstHPOsearchspace)'''

                    print("==", _thisStepsize, " runs on the whole search space")
                    _imax_eval = _imax_eval + _thisStepsize
                    _eval_counted += _thisStepsize
                    _max_eval = _max_eval + _thisStepsize
                    kwargs['max_queue_len'] = min(_thisStepsize, self.hpo_max_queue_len)
                    kwargs['search_space'] = search_space
                    kwargs['max_evals'] = _max_eval
                    kwargs['fix_max_evals'] = _imax_eval
                    kwargs['n_init_sample'] = _imax_eval
                    kwargs['timeout'] = self.timeout - (
                            time.time() - self.start_time) if self.timeout != None else None
                    if self.hpo_suggest == "tpe":
                        _hpo_algo = partial(tpe.suggest, n_startup_jobs=_max_eval,
                                            n_EI_candidates=self.n_EI_candidates)
                        kwargs['algo'] = _hpo_algo
                    # kwargs['algo'] = rand.suggest
                    BO = HO.HyperOpt(**kwargs)
                    # print("====MAX:", _max_eval)
                    _, _, _max_eval, _imax_eval = BO.run()
                print("END Random search on ",len(self._LstHPOsearchspace)," combinations")
                del self._LstHPOsearchspace
                if self.hpo_suggest == "tpe":
                    _hpo_algo = partial(tpe.suggest, n_startup_jobs=_max_eval,
                                        n_EI_candidates=self.n_EI_candidates)
                    kwargs['algo'] = _hpo_algo
                kwargs['max_queue_len'] = self.hpo_max_queue_len
                #_max_eval =self.max_eval+(_max_eval-_imax_eval)
                kwargs['search_space'] = search_space
                kwargs['max_evals'] = self.max_eval if self.max_eval==-1 else self.max_eval+(_max_eval-_imax_eval)
                kwargs['timeout'] = self.timeout - (time.time() - self.start_time) if self.timeout!=None else None
                kwargs['fix_max_evals'] = self.max_eval
                kwargs['isParallel'] = False
                kwargs['trials']=self.trials

                BO = HO.HyperOpt(**kwargs)

            else:
                kwargs['search_space'] = search_space
                kwargs['fix_max_evals'] = self.max_eval
                kwargs['timeout'] = self.timeout - (time.time() - self.start_time)if self.timeout!=None else None
                BO = HO.HyperOpt(**kwargs)
        else:
            pass
        _return= BO.run()
        if (self.isHyperopt):
            _trials = sorted([x for x in BO.trials], key=lambda x: x["book_time"])
            self.results=self._save_results(_trials)
            self.trials=_trials
            #del _trials
        return _return

if __name__ == '__main__':
    print('NONE')

