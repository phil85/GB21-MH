
# import algorithm
from algorithm import gb21_mh

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

# apply algorithm
medians, assignments = gb21_mh(inst_path, t_total,
                               n_start, g_initial, init,
                               n_target, l, t_local,
                               mip_gap_global, mip_gap_local,
                               np_seed, gurobi_seed)
