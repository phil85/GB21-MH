# GB21-MH

Clustering algorithm for large-scale capacitated clustering. 

## Dependencies

GB21-MH depends on:
* [Gurobi](https://anaconda.org/Gurobi/gurobi)
* [Numpy](https://anaconda.org/conda-forge/numpy)
* [Scipy](https://anaconda.org/anaconda/scipy)
* [Scikit Learn](https://anaconda.org/anaconda/scikit-learn)
* [Time](https://anaconda.org/conda-forge/time)

Gurobi is a commercial mathematical programming solver. Free academic licenses are available [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Installation

1) Download and install Gurobi (https://www.gurobi.com/downloads/)
2) Download and install Git Large File Storage (https://git-lfs.github.com/)
3) Clone this repository (git clone https://github.com/phil85/GB21-MH.git)

## Usage

The main.py file contains code that applies the GB21-MH algorithm on an illustrative example.

```python
medians, assignments = gb21_mh(X, Q, q, p, t_total, n_start, g_initial, init, n_target, l, t_local, flag_local, mip_gap_global, mip_gap_local, np_seed, gurobi_seed)
```

Instance:
* X: np.array, feature vectors of objects
* Q: np.array, capacities of objects
* q: np.array, weights of objects
* p: int, number of clusters

Tuning parameters of algorithm:
* t_total: float, time limit on total running time
* n_start: int, number of runs of global optimization phase  
* g_initial: int, initial number of nearest medians to which an object can be assigned
* init: str, initialization method to determine initial set of medians
* n_target: int, target number of objects in initial subset
* l: int, number of nearest objects to each median to be considered as potential new medians 
* t_local: float, time limit for solving model in local optimization phase
* flag_local: boolean, no local optimization phase if set to false
* mip_gap_global: float, additional termination criterion for solving model in global optimization phase
* mip_gap_local: float, additional termination criterion for solving model in local optimization phase

Random seeds:
* np_seed: int, random seed for numpy
* gurobi_seed: int, random seed for gurobi solver

For a more detailed description, please refer to the paper below.

## Reference

Please cite the following paper if you use the algorithm or the instances.

**Gnägi, M., Baumann, P.** (2021): A matheuristic for large-scale capacitated clustering. Computers & Operations Research. To appear

Bibtex:
```
@article{Gna21,
  title={A matheuristic for large-scale capacitated clustering},
  author={Gn{\"a}gi, Mario and Baumann, Philipp},
  journal={Computers \& Operations Research},
  volume={},
  pages={},
  year={},
  publisher={Elsevier},
  note={To appear}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


