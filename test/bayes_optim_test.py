from bayes_optim import BO, ContinuousSpace, NominalSpace, OrdinalSpace
import numpy as np
from bayes_optim.Surrogate import GaussianProcess, RandomForest

dim = 5
# Define Search Space
alg_namestr = NominalSpace(["SVM"], "alg_namestr")

# Define Search Space for Support Vector Machine
kernel = NominalSpace(["linear", "rbf"], "kernel")
test = NominalSpace(["A", "B"], "test")
C = ContinuousSpace([1e-2, 100], "C")
degree = OrdinalSpace([1, 5], 'degree')
#space = ContinuousSpace([-5, 5]) * dim  # create the search space
space=kernel+test+C +degree
def new_obj(params):
    print(params)
    return (np.random.uniform(0, 1))

model = RandomForest(levels=space.levels)
opt = BO(
    search_space=space,
    obj_fun=new_obj,
    model=model,
    DoE_size=5,                         # number of initial sample points
    max_FEs=50,                         # maximal function evaluation
    verbose=False,
    eval_type='dict'
)
opt.run()