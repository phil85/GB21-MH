
# import algorithm
from algorithm import gb21_mh
from algorithm import read_inst

# package for visualisation
import matplotlib.pyplot as plt

# path of instance file
inst_path = './illustrative_example.txt'

# maximum time limit
t_total = 3600

# input parameters
n_start = 1
g_initial = 3
init = "kmeans++"  # alternatively: "capacity-based"
n_target = 7
l = 3
t_local = 300
mip_gap_global = 0.01
mip_gap_local = 0.01

# random seeds
np_seed = 4
gurobi_seed = 4

# additional parameters
no_local = False
no_solver = False

# read test instance from txt-file
X, Q, q, p = read_inst(inst_path)

# apply algorithm
medians, assignments = gb21_mh(X, Q, q, p, t_total,
                               n_start, g_initial, init,
                               n_target, l, t_local,
                               mip_gap_global, mip_gap_local,
                               np_seed, gurobi_seed,
                               no_local, no_solver)

# plot solution (if two-dimensional)
if X.shape[1] == 2:
    for i in range(X.shape[0]):
        j = assignments[i]
        plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color='black')
        plt.scatter(X[i, 0], X[i, 1], color='grey')

plt.show()
