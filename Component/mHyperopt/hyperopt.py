import hyperopt
class HyperOpt(object):
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self,round_id=0):
        self.fmin = hyperopt.fmin(fn=self.obj_func, space=self.search_space, algo=self.algo,
                                 max_evals=self.max_eval, trials=self.trials, rstate=self.rstate,
                                 pass_expr_memo_ctrl=self.pass_expr_memo_ctrl, verbose=self.verbose,
                                 return_argmin=self.return_argmin, max_queue_len=self.max_queue_len,
                                 show_progressbar=self.show_progressbar)
        xcatch= [x['loss'] for x in self.trials.results]
        self.fopt = min(xcatch)
        self.ieval_count=len(xcatch)
        return self.fmin,self.fopt,None, self.ieval_count
    def AddBudget_run(self,add_eval, round_id=1):
        self.max_eval += add_eval
        return self.run(round_id)
