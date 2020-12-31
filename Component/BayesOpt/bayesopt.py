from __future__ import absolute_import
from . import BO
from . import RandomForest
import BanditOpt.ParamExtension as ext

import types
class BayesOpt(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if (self.model == None):
            model = RandomForest(levels=self.search_space.levels)
        else:
            model = self.surrogate
        '''max_eval=self.max_eval,  # we evaluate maximum n_init_sample times
                                         max_iter=self.max_iter,  # we have max 500 iterations
                                         init_points=self.init_points,

                                         infill=self.infill,  # Expected improvement as criteria
                                         noisy=self.noisy, t0=self.t0, tf=self.tf,schedule=self.schedule,

                                         n_init_sample=self.n_init_sample,  # We start with 10 initial samples
                                         #n_restart=self.n_restart,
                     #max_infill_eval=self.max_infill_eval,
                     #wait_iter=self.wait_iter,
                     #optimizer=self.optimizer,  # We use the MIES internal optimizer.
        '''
        #Please check the detail: https://github.com/wangronin/Bayesian-Optimization/blob/6741c05e336ed15970c1d7bf5a38f86e6ebada00/bayes_optim/base.py#L178

        self.BO = BO(search_space=self.search_space,
                     obj_fun=self.obj_fun,
                     #surrogate=model,
                     parallel_obj_fun=self.parallel_obj_fun,
                     eq_fun=self.eq_fun,
                     ineq_fun=self.ineq_fun,
                     model=model, #new
                     eval_type=self.eval_type,  # use this parameter to control the type of evaluation
                     DoE_size=self.DoE_size, #new
                     warm_data=self.warm_data,
                     n_point=self.n_point,  # We evaluate every iteration 1 time
                     acquisition_fun = self.acquisition_fun, #new
                     acquisition_par=self.acquisition_par,#new
                     acquisition_optimization= self.acquisition_optimization,#new
                     ftarget=self.ftarget,
                     max_FEs=self.max_FEs,#new
                     minimize=self.minimize,  # the problem is a minimization problem.
                     n_job=self.n_job,  # with 1 process (job).
                     data_file=self.data_file,
                     verbose=self.verbose,
                     random_seed=self.random_seed,
                     logger=self.logger)
        funcType = types.MethodType
        self.BO.pre_eval_check = funcType(ext.check_configuration,self.BO)
        if(self.eval_type=="dict"):
            #D.A.Nguyen: 31.12.2020 Hide this row
            #self.BO.run = funcType(ext.BOrun_todict,self.BO)
            self.BO.evaluate = funcType(ext.evaluate,self.BO)
            self.BO.formatCandidate =funcType(ext.formatCandidate,self.BO)
        self.BO._forbidden=self.forbidden
        self.BO._conditional=self.conditional
        self.BO._isBandit=self.isBandit
        self.BO._hyperparameters=self.hyperparameters
    def run(self, round_id=0):
        sp_id=self.sp_id
        #xopt, fitness, stop_dict = self.BO.run()
        try:
            xopt, fitness, stop_dict = self.BO.run()
        except Exception as e:
            xopt = None
            stop_dict = None
            try:
                fitness=min(self.BO.hist_f)
            except Exception as exceptMsg:
                if(self.minimize):
                    fitness=1
                else:
                    fitness=0
                print(exceptMsg)
            print('Round ',str(round_id),' ERROR:', str(sp_id), '--msg:', e)
        #iter_count=self.BO.iter_count
        eval_count=self.BO.eval_count
        return xopt, fitness, stop_dict, eval_count
    def AddBudget_run(self, add_eval, round_id=1):
        self.BO.n_init_sample=0
        current_Max_eval=max(self.BO.max_FEs,self.BO.eval_count)
        self.BO.max_FEs=current_Max_eval+add_eval
        self.BO.stop_dict.clear()
        return self.run(round_id)


