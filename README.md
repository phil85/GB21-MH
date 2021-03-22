# GB21-MH

Clustering algorithm for large-scale capacitated clustering. 

## Dependencies

GB21-MH depends on:
* [Gurobi](https://anaconda.org/Gurobi/gurobi)
* [Numpy](https://anaconda.org/conda-forge/numpy)
* [Scipy](https://anaconda.org/anaconda/scipy)

Gurobi is a commercial mathematical programming solver. Free academic licenses are available [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Installation

1) Download and install Gurobi (https://www.gurobi.com/downloads/)
2) Download and install [Git Large File Storage (LFS)](https://git-lfs.github.com/) (git lfs install)
3) Clone this repository (git clone https://github.com/phil85/GB21-MH.git)

## Usage

The main.py file contains code that applies the GB21-MH algorithm on an illustrative example.

```python
labels = gb21_mh(tbd)
```

## Reference

Please cite the following paper if you use this algorithm.

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


