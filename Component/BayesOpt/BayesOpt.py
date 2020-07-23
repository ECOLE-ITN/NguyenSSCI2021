from BayesOpt import BO
from BayesOpt.Surrogate import RandomForest
from BayesOpt.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace
import BanditOpt.ParamExtension as ext
import types
class BayesOpt(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if (self.surrogate == None):
            model = RandomForest(levels=self.search_space.levels)
        else:
            model = self.surrogate

        self.BO = BO(self.search_space, self.obj_func, surrogate=model, ftarget=self.ftarget,
                                 parallel_obj_func=self.parallel_obj_func,
                                 eq_func=self.eq_func,
                                 ineq_func=self.ineq_func,
                                 minimize=self.minimize,  # the problem is a minimization problem.
                                 max_eval=self.max_eval,  # we evaluate maximum n_init_sample times
                                 max_iter=self.max_iter,  # we have max 500 iterations
                                 init_points=self.init_points,
                                 warm_data=self.warm_data,
                                 infill=self.infill,  # Expected improvement as criteria
                                 noisy=self.noisy, t0=self.t0, tf=self.tf,schedule=self.schedule,
                                 eval_type=self.eval_type, # use this parameter to control the type of evaluation
                                 n_init_sample=self.n_init_sample,  # We start with 10 initial samples
                                 n_point=self.n_point,  # We evaluate every iteration 1 time
                                 n_job=self.n_job,  # with 1 process (job).
                                 n_restart=self.n_restart, max_infill_eval=self.max_infill_eval,
                                 wait_iter=self.wait_iter,
                                 optimizer=self.optimizer,  # We use the MIES internal optimizer.
                                 data_file=self.data_file,
                                 verbose=self.verbose, random_seed=self.random_seed,
                                 logger=self.logger)
        funcType = types.MethodType
        self.BO.pre_eval_check = funcType(ext.check_configuration,self.BO)
        if(self.eval_type=="dict"):
            self.BO.run = funcType(ext.BOrun_todict,self.BO)
        self.BO._forbidden=self.forbidden
        self.BO._conditional=self.conditional
        self.BO._isBandit=self.isBandit
        self.BO._hyperparameters=self.hyperparameters
    def run(self, round_id=0):
        sp_id=self.sp_id

        try:
            xopt, fitness, stop_dict = self.BO.run()
        except Exception as e:
            xopt = None
            stop_dict = None
            try:
                fitness=min(self.BO.eval_hist)
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
        current_Max_eval=max(self.BO.max_eval,self.BO.eval_count)
        self.BO.max_eval=current_Max_eval+add_eval
        self.BO.stop_dict.clear()
        return self.run(round_id)


