from collections import OrderedDict
from math import log, ceil, floor

import numpy as np

from mipego4ml.ConditionalSpace import ConditionalSpace
from mipego4ml.ConfigSpace import ConfigSpace
from mipego4ml.mipego.Surrogate import RandomForest
from mipego4ml.mipego.mipego import mipego


class mipego4ML(object):
    def __init__(self, search_space: ConfigSpace, conditional: ConditionalSpace, obj_func, eta=3, ftarget=None,
                 minimize=True, noisy=False, max_eval=None, max_iter=None,
                 infill='EI', t0=2, tf=1e-1, schedule=None,
                 n_init_sample=None, n_point=1, n_job=1, backend='multiprocessing',
                 n_restart=None, max_infill_eval=None, wait_iter=3, optimizer='MIES',
                 log_file=None, data_file=None, verbose=False, random_seed=None,
                 available_gpus=[]):
        self.eta = eta
        self.obj_func = obj_func
        self.minimize = minimize
        self.noisy = noisy
        self.infill = infill
        self.t0 = t0
        self.tf = tf
        self.schedule = schedule
        self.n_init_sample = n_init_sample
        self.n_point = n_point
        self.n_job = n_job
        self.backend = backend
        self.n_restart = n_restart
        self.max_infill_eval = max_infill_eval
        self.wait_iter = wait_iter
        self.optimizer = optimizer
        self.log_file = log_file
        self.data_file = data_file
        self.verbose = verbose
        self.random_seed = random_seed
        self.available_gpus = available_gpus
        self.max_eval = max_eval
        self.max_iter = max_iter
        self.eta = 3
        self.iter_count = 0
        self.eval_count = 0
        self.ftarget = ftarget
        self.stop_dict = {}
        self.searchspace = search_space.combinewithconditional(conditional)
        self.gb_N_value_count = sum([len(search_space._OrgLevels[i]) for i in search_space._OrgLevels])
        # print(gb_N_value_count)
        # lsRunning = OrderedDict()
        self._lsCurrentBest = OrderedDict()
        self._lsOrderdBest = OrderedDict()
        self._lsincumbent = OrderedDict()
        self.opt = OrderedDict()

    def run(self):
        print('MIPEGO4ML--Version:1.0.2')
        imax_eval = 0
        imax_iter = 0

        for sp in self.searchspace:
            sp_id = self.searchspace.index(sp)
            model = RandomForest(levels=sp.levels)
            self.opt[sp_id] = mipego(sp, self.obj_func, surrogate=model, ftarget=self.ftarget,
                                         minimize=self.minimize,  # the problem is a minimization problem.
                                         max_eval=self.n_init_sample,  # we evaluate maximum 500 times
                                         # max_iter=self.imax_iter,  # we have max 500 iterations
                                         infill=self.infill,  # Expected improvement as criteria
                                         n_init_sample=self.n_init_sample,  # We start with 10 initial samples
                                         n_point=self.n_point,  # We evaluate every iteration 1 time
                                         n_job=self.n_job,  # with 1 process (job).
                                         optimizer=self.optimizer,  # We use the MIES internal optimizer.
                                         verbose=self.verbose, random_seed=self.random_seed, noisy=self.noisy,
                                         t0=self.t0, tf=self.tf, schedule=self.schedule, backend=self.backend,
                                         n_restart=self.n_restart, max_infill_eval=self.max_infill_eval,
                                         wait_iter=self.wait_iter,
                                         log_file=self.log_file, data_file=self.data_file,
                                         available_gpus=self.available_gpus)
            try:
                incumbent, stop_dict = self.opt[sp_id].run()
                self._lsincumbent[sp_id] = incumbent
                self._lsCurrentBest[sp_id] = min(self.opt[sp_id].eval_hist)
                print('INIT message:',str(sp_id), '--best: ',str(min(self.opt[sp_id].eval_hist)))
            except Exception as e:
                print('INIT Round ERROR:',str(sp_id),'--msg:',e)
            self.iter_count += self.opt[sp_id].iter_count
            self.eval_count += self.opt[sp_id].eval_count
            # lsRunning[sp_id] = [incumbent,model,stop_dict, opt[sp_id]]
        max_eval = self.max_eval - self.eval_count
        max_iter = self.max_iter - self.iter_count
        lsRace = self.calculateSH()
        num_races = len(lsRace)
        eval_race = max_eval / num_races
        iter_race = max_iter / num_races
        best_incumbent = ""
        best_value = 0.000000
        errList=[]
        for iround, num_candidate in lsRace.items():
            print("The ", iround + 1, " round, Runing:", num_candidate, "Candidates")
            if (self.minimize == True):
                lsThisRound = list(OrderedDict(sorted(self._lsCurrentBest.items(), key=lambda item: item[1])).items())[
                              :num_candidate]
                #self._lsCurrentBest=OrderedDict(lsThisRound)
            else:
                lsThisRound = list(OrderedDict(sorted(self._lsCurrentBest.items(), key=lambda item: item[1],
                                                      reverse=True)).items())[:num_candidate]
            for cdid, bestloss in lsThisRound:
                cd_add_eval = int(floor(eval_race / num_candidate))
                cd_add_iter = int(floor(iter_race / num_candidate))
                if (num_candidate<=1):
                    remain_eval = self.max_eval - self.eval_count
                    remain_iter = self.max_iter - self.iter_count
                    cd_add_eval = max(cd_add_eval,remain_eval)
                    cd_add_iter = max(cd_add_iter,remain_iter)

                # lc_N_value_count= [len(sp.le)]
                print("previous best loss was:", bestloss, "of", cdid)
                # cdvalue
                cd_ran_iter= self.opt[cdid].iter_count
                cd_ran_eval= self.opt[cdid].eval_count
                self.opt[cdid].n_init_sample = 0
                self.opt[cdid].max_iter = max(self.opt[cdid].max_iter,self.opt[cdid].iter_count) + cd_add_iter
                self.opt[cdid].max_eval = max(self.opt[cdid].max_eval,self.opt[cdid].eval_count) + cd_add_eval
                self.opt[cdid].stop_dict.clear()
                try:
                    incumbent, stop_dict = self.opt[cdid].run()
                except Exception as e:
                    self.opt[cdid].max_iter=self.opt[cdid].iter_count
                    self.opt[cdid].max_eval=self.opt[cdid].eval_count
                    cd_add_iter = self.opt[cdid].iter_count - cd_ran_iter
                    cd_add_eval = self.opt[cdid].eval_count - cd_ran_eval
                    if(cdid not in errList):
                        errList.append(cdid)
                    print('MIPEGO-ERROR:',e)
                #update infor
                self.iter_count += cd_add_iter
                self.eval_count += cd_add_eval
                self._lsincumbent[cdid] = incumbent
                best_incumbent = incumbent
                best_value = self.opt[cdid].eval_hist[self.opt[cdid].incumbent_id]
                self._lsCurrentBest[cdid] = best_value
                #self._lsCurrentBest[cdid] = np.mean(self.opt[cdid].eval_hist)
                #self._lsCurrentBest[cdid] = np.mean(self.opt[cdid].eval_hist[-cd_add_iter:])
        errIDs= errList
        print('145:',len(errIDs))
        while (len(errList)>0):
            if(len(errIDs)>=len(self.searchspace)):
                break
            print("Additional round, Runing: 1 Candidate")
            shortLst=OrderedDict([f for f in self._lsCurrentBest.items() if f[0] not in errIDs])
            errList=[]
            if (self.minimize == True):
                lsThisRound = list(OrderedDict(sorted(shortLst.items(), key=lambda item: item[1])).items())[:1]
                #self._lsCurrentBest=OrderedDict(lsThisRound)
            else:
                lsThisRound = list(OrderedDict(sorted(shortLst.items(), key=lambda item: item[1],
                                                      reverse=True)).items())[:1]
            for cdid, bestloss in lsThisRound:
                remain_eval = self.max_eval - self.eval_count
                remain_iter = self.max_iter - self.iter_count
                cd_add_eval = remain_eval
                cd_add_iter = remain_iter
                # lc_N_value_count= [len(sp.le)]
                print("previous best loss was:", bestloss, "of", cdid)
                # cdvalue
                cd_ran_iter= self.opt[cdid].iter_count
                cd_ran_eval= self.opt[cdid].eval_count
                self.opt[cdid].n_init_sample = 0
                self.opt[cdid].max_iter = max(self.opt[cdid].max_iter,self.opt[cdid].iter_count) + cd_add_iter
                self.opt[cdid].max_eval = max(self.opt[cdid].max_eval,self.opt[cdid].eval_count) + cd_add_eval
                self.opt[cdid].stop_dict.clear()
                try:
                    incumbent, stop_dict = self.opt[cdid].run()
                except Exception as e:
                    self.opt[cdid].max_iter=self.opt[cdid].iter_count
                    self.opt[cdid].max_eval=self.opt[cdid].eval_count
                    cd_add_iter = self.opt[cdid].iter_count - cd_ran_iter
                    cd_add_eval = self.opt[cdid].eval_count - cd_ran_eval
                    if(cdid not in errIDs):
                        errIDs.append(cdid)
                    errList.append(cdid)
                    print(e)
                #update infor
                self.iter_count += cd_add_iter
                self.eval_count += cd_add_eval
                self._lsincumbent[cdid] = incumbent
                best_incumbent = incumbent
                best_value = self.opt[cdid].eval_hist[self.opt[cdid].incumbent_id]
                self._lsCurrentBest[cdid] = best_value
        #conclusion
        if (self.minimize == True):
            lsThisRound = list(OrderedDict(sorted(self._lsCurrentBest.items(), key=lambda item: item[1])).items())[:1]
            #self._lsCurrentBest=OrderedDict(lsThisRound)
        else:
            lsThisRound = list(OrderedDict(sorted(self._lsCurrentBest.items(), key=lambda item: item[1],
                                                  reverse=True)).items())[:1]
        best_cdid = lsThisRound[0][0]
        best_incumbent,best_value=self._lsincumbent[best_cdid], self._lsCurrentBest[best_cdid]
        return best_incumbent, best_value

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
    max_iter = 81  # maximum iterations per configuration
    eta = 3  # defines configuration downsampling rate (default = 3)

    logeta = lambda x: log(x) / log(eta)
    s_max = int(logeta(max_iter))
    B = (s_max + 1) * max_iter

    results = []  # list of dicts
    counter = 0
    best_loss = np.inf
    best_counter = -1
