from . import hyperopt, tpe, space_eval
from functools import partial
import time
class HyperOpt(object):
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.def_max_fails=10
        self.fmin_todict= None

    def run(self,round_id=0):
        _org_max_eval = self.fix_max_evals
        _this_max_eval=self.max_evals
        _stattime=time.time()
        if self.algo_str=='tpe':
            _initMax_evals=self.max_evals
            try:
                self.fmin = hyperopt.fmin(fn=self.obj_func, space=self.search_space, algo=self.algo,
                                          max_evals=_initMax_evals, trials=self.trials, rstate=self.rstate,
                                          pass_expr_memo_ctrl=self.pass_expr_memo_ctrl, verbose=self.verbose,
                                          return_argmin=self.return_argmin, max_queue_len=self.max_queue_len,timeout=self.timeout,
                                          show_progressbar=self.show_progressbar)
            except:
                pass
            xcatch = [x['loss'] for x in self.trials.results if x['status'] == 'ok']
            ieval_count = len(xcatch)
            _newInit = 0
            self.timeout=self.timeout-(time.time()-_stattime) if self.timeout!=None else None
            #if ieval_count < self.n_init_sample:
            while ieval_count < self.n_init_sample and (self.timeout!=None and self.timeout>0):
                eval_count = len([x['loss'] for x in self.trials.results])
                _initMax_evals += (self.n_init_sample - ieval_count)
                _lastresults = [x for x in self.trials.results[-self.def_max_fails:] if x["status"] == "ok"]
                # print(self.def_max_fails-len(_lastresults))
                if len(_lastresults) == 0 and eval_count > _this_max_eval + self.def_max_fails:
                    print("too many fails, stop mining on this area")
                    break
                self.algo = partial(tpe.suggest, n_startup_jobs=_initMax_evals)
                try:
                    self.fmin = hyperopt.fmin(fn=self.obj_func, space=self.search_space, algo=self.algo,
                                              max_evals=_initMax_evals, trials=self.trials, rstate=self.rstate,
                                              pass_expr_memo_ctrl=self.pass_expr_memo_ctrl, verbose=self.verbose,
                                              return_argmin=self.return_argmin, max_queue_len=self.max_queue_len,timeout=self.timeout,
                                              show_progressbar=self.show_progressbar)
                except:
                    pass
                ieval_count = len([x['loss'] for x in self.trials.results if x['status'] == 'ok'])
                self.timeout = self.timeout - (time.time() - _stattime) if self.timeout != None else None
            #start BO
            '''ieval_count = len([x['loss'] for x in self.trials.results if x['status'] == 'ok'])
            eval_count = len([x['loss'] for x in self.trials.results])
            _addeval = eval_count - ieval_count
            self.max_evals += _addeval
            self.fmin = hyperopt.fmin(fn=self.obj_func, space=self.search_space, algo=self.algo,
                                      max_evals=self.max_evals, trials=self.trials, rstate=self.rstate,
                                      pass_expr_memo_ctrl=self.pass_expr_memo_ctrl, verbose=self.verbose,
                                      return_argmin=self.return_argmin, max_queue_len=self.max_queue_len,
                                      show_progressbar=self.show_progressbar)'''
        else:
            try:
                self.fmin = hyperopt.fmin(fn=self.obj_func, space=self.search_space, algo=self.algo,
                                      max_evals=self.max_evals, trials=self.trials, rstate=self.rstate,
                                      pass_expr_memo_ctrl=self.pass_expr_memo_ctrl, verbose=self.verbose,
                                      return_argmin=self.return_argmin, max_queue_len=self.max_queue_len,
                                      timeout=self.timeout, show_progressbar=self.show_progressbar)
            except:
                pass
        xcatch= [x['loss'] for x in self.trials.results if x['status']=='ok' ]
        ieval_count = len(xcatch)
        self.timeout = self.timeout - (time.time() - _stattime) if self.timeout != None else None
        while ieval_count<_org_max_eval and (self.timeout!=None and self.timeout>0):
            _lastresults = [x for x in self.trials.results[-self.def_max_fails:] if x["status"] == "ok"]
            #print(self.def_max_fails - len(_lastresults), ' FAILS')
            eval_count = len([x['loss'] for x in self.trials.results])
            if len(_lastresults) == 0 and eval_count>_this_max_eval+self.def_max_fails:
                print("too many fails, stop mining on this area")
                break
            _addeval=eval_count-ieval_count
            self.max_evals=_org_max_eval+_addeval
            try:
                self.fmin = hyperopt.fmin(fn=self.obj_func, space=self.search_space, algo=self.algo,
                                          max_evals=self.max_evals, trials=self.trials, rstate=self.rstate,
                                          pass_expr_memo_ctrl=self.pass_expr_memo_ctrl, verbose=self.verbose,
                                          return_argmin=self.return_argmin, max_queue_len=self.max_queue_len,
                                          timeout=self.timeout,show_progressbar=self.show_progressbar)
            except:
                pass
            ieval_count=len([x['loss'] for x in self.trials.results if x['status'] == 'ok'])
            self.timeout = self.timeout - (time.time() - _stattime) if self.timeout != None else None
        xcatch = [x['loss'] for x in self.trials.results if x['status'] == 'ok']
        ieval_count = len(xcatch)
        self.fopt = self.trials.best_trial['result']['loss']
        self.eval_count = len([x['loss'] for x in self.trials.results])
        self.ieval_count=ieval_count
        self.eval_hist=xcatch
        if not hasattr(self, 'fmin'):
            self.fmin={k:v[0] for k,v in self.trials.best_trial['misc']['vals'].items() if len(v)>0}

        self.fmin_todict = space_eval(self.search_space, {k: v for k, v in self.fmin.items()})
        if hasattr(self,"isParallel"):
            if self.isParallel==True:
                return self.fmin_todict,self.fopt,self.eval_count, self.ieval_count,self.trials, self.sp_id, self.rstate

        return self.fmin_todict,self.fopt,self.eval_count, self.ieval_count
    def AddBudget_run(self,add_eval, round_id=1):
        try:
            self.max_evals=len([x['loss'] for x in self.trials.results])
        except:
            pass
        self.max_evals += add_eval
        return self.run(round_id)

