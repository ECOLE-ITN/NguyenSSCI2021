from Component.mHyperopt import hp
from BanditOpt.ConfigSpace import ConfigSpace, SearchSpace
import copy
def ToHyperopt(search_space: list):
    if(len(search_space)<=1):
        print("Please use hyperopt directly")
        return
    else:
        Ls_hpOpt=[]
        for ls in search_space:
            hOpt = dict()
            varname = ls.var_name
            levels = list(ls.levels.items())
            bounds = ls.bounds
            vartype = ls.var_type
            i = 0
            while i < len(varname):
                if (i in ls.id_N):  # Categorical
                    hOpt[varname[i]] = hp.choice(varname[i], (([a for a in bounds[i]])))
                elif (i in ls.id_C):  ##real number
                    hOpt[varname[i]] = hp.uniform(varname[i], float(bounds[i][0]), float(bounds[i][1]))
                elif (i in ls.id_O):  ##interge
                    hOpt[varname[i]] = hp.choice(varname[i], range(int(bounds[i][0]), int(bounds[i][1])))
                i = i + 1
            # print('lstparams.append(cs)')
            Ls_hpOpt.append(copy.deepcopy(hOpt))
        return Ls_hpOpt