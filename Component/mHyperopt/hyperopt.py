from . import hyperopt, tpe, space_eval, rand
from functools import partial
import time
class HyperOpt(object):
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.def_max_fails=10
        self.fmin_todict= None
        self.isInitMode=True

    def run(self,round_id=0):
        self.isError = False
        _org_max_eval = self.fix_max_evals
        _this_max_eval=self.max_evals
        try:
            ieval_count = len([x['loss'] for x in self.trials.results if x['status'] == 'ok'])
            eval_count = len([x['loss'] for x in self.trials.results])
        except:
            ieval_count,eval_count=0,0
        _stattime=time.time()
        if self.algo_str=='tpe':
            self.isInitMode = False if ieval_count >= self.n_init_sample else True
            if self.isInitMode==True:
                #print('****INIT >>>>>')
                #_initStep = self.n_init_sample + ((eval_count-ieval_count) if self.isInitMode else 0)
                _errorCount = 0
                while (_org_max_eval>ieval_count and (self.timeout>0 if self.timeout!=None else True)):
                    #_initStep = self.n_init_sample + (_org_max_eval - _this_max_eval)
                    _algo=rand.suggest
                    _initMax_evals=eval_count+(_org_max_eval-ieval_count)
                    try:
                        self.fmin = hyperopt.fmin(fn=self.obj_func, space=self.search_space, algo=_algo,
                                                  max_evals=_initMax_evals, trials=self.trials, rstate=self.rstate,
                                                  pass_expr_memo_ctrl=self.pass_expr_memo_ctrl, verbose=self.verbose,
                                                  return_argmin=self.return_argmin, max_queue_len=self.max_queue_len,
                                                  timeout=self.timeout,
                                                  show_progressbar=self.show_progressbar)
                    except Exception as e:
                        print(e)
                        _errorCount +=1
                        pass
                    ieval_count = len([x['loss'] for x in self.trials.results if x['status'] == 'ok'])
                    eval_count = len([x['loss'] for x in self.trials.results])
                    #_initStep = self.n_init_sample + (_org_max_eval - _this_max_eval)
                    _lastresults = [x for x in self.trials.results[-self.def_max_fails:] if x["status"] == "ok"]
                    self.isInitMode = False if ieval_count >= self.n_init_sample else True
                    if (eval_count > _this_max_eval + self.def_max_fails) or _errorCount>self.def_max_fails :
                        print("Hyperopt message: too many fails, stop mining on this area")
                        self.isError = True
                        break
                    if self.isInitMode==False:
                        print("Hyperopt message: Finish Random Mode => Move to BO Mode*****")
                        break
            else:
                # If not Initial sampling
                #print('****BO >>>>>')
                _errorCount=0
                while ieval_count < _org_max_eval and (self.timeout > 0 if self.timeout != None else True):
                    _addeval = _org_max_eval - ieval_count
                    _initMax_evals =eval_count+ _addeval
                    try:
                        self.fmin = hyperopt.fmin(fn=self.obj_func, space=self.search_space, algo=self.algo,
                                                  max_evals=_initMax_evals, trials=self.trials, rstate=self.rstate,
                                                  pass_expr_memo_ctrl=self.pass_expr_memo_ctrl, verbose=self.verbose,
                                                  return_argmin=self.return_argmin, max_queue_len=self.max_queue_len,
                                                  timeout=self.timeout, show_progressbar=self.show_progressbar)
                        _errorCount =0
                    except Exception as e:
                        _errorCount+=1
                        print(e)
                        pass
                    ieval_count = len([x['loss'] for x in self.trials.results if x['status'] == 'ok'])
                    #_lastresults = [x for x in self.trials.results[-self.def_max_fails:] if x["status"] == "ok"]
                    eval_count = len([x['loss'] for x in self.trials.results])
                    #print(eval_count, _this_max_eval )
                    if _errorCount>self.def_max_fails*2:
                        print("too many fails, stop mining on this area", _errorCount)
                        self.isError = True
                        break
                    self.timeout = self.timeout - (time.time() - _stattime) if self.timeout != None else None
        else:
            try:
                self.fmin = hyperopt.fmin(fn=self.obj_func, space=self.search_space, algo=self.algo,
                                      max_evals=self.max_evals, trials=self.trials, rstate=self.rstate,
                                      pass_expr_memo_ctrl=self.pass_expr_memo_ctrl, verbose=self.verbose,
                                      return_argmin=self.return_argmin, max_queue_len=self.max_queue_len,
                                      timeout=self.timeout, show_progressbar=self.show_progressbar)
            except Exception as e:
                print(e)
                pass
        self.eval_count = len([x['loss'] for x in self.trials.results])
        self.ieval_count = len([x['loss'] for x in self.trials.results if x['status'] == 'ok'])
        self.eval_hist = self.ieval_count
        try:
            self.fopt = self.trials.best_trial['result']['loss']
            #print('_org_max_eval: ',_org_max_eval,' ieval_count:',ieval_count,' eval_count:',eval_count)
            if not hasattr(self, 'fmin'):
                self.fmin={k:v[0] for k,v in self.trials.best_trial['misc']['vals'].items() if len(v)>0}
            try:
                self.fmin_todict = space_eval(self.search_space, {k: v for k, v in self.fmin.items()})
            except Exception as e:
                self.fmin_todict={}
                print(e)
            if hasattr(self,"isParallel"):
                if self.isParallel==True:
                    return self.fmin_todict,self.fopt,self.eval_count, self.ieval_count#self.trials, self.sp_id, self.rstate

            return self.fmin_todict,self.fopt,self.eval_count, self.ieval_count
        except Exception as e:
            print('An Unknown error: ', e)
            return {}, 1, self.eval_count, self.ieval_count
    def AddBudget_run(self,add_eval, round_id=1):
        try:
            self.fix_max_evals=len([x['loss'] for x in self.trials.results if x['status'] == 'ok'])
            self.max_evals=len([x['loss'] for x in self.trials.results])
        except:
            pass
        self.fix_max_evals +=add_eval
        self.max_evals += add_eval
        return self.run(round_id)

