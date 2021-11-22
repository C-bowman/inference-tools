# inference-tools

[![Documentation Status](https://readthedocs.org/projects/inference-tools/badge/?version=stable)](https://inference-tools.readthedocs.io/en/stable/?badge=stable)
[![GitHub license](https://img.shields.io/github/license/C-bowman/inference-tools?color=blue)](https://github.com/C-bowman/inference-tools/blob/master/LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/inference-tools?color=purple)](https://pypi.org/project/inference-tools/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/inference-tools)
[![DOI](https://zenodo.org/badge/149741362.svg)](https://zenodo.org/badge/latestdoi/149741362)

This package provides a set of Python-based tools for Bayesian data analysis
which are simple to use, allowing them to applied quickly and easily.

Inference-tools is not a framework for Bayesian modelling (e.g. like [PyMC](https://docs.pymc.io/)),
but instead provides tools to sample from user-defined models using MCMC, and to analyse and visualise
the sampling results.

## Features

 - Implementations of MCMC algorithms like Gibbs sampling and Hamiltonian Monte-Carlo for 
 sampling from user-defined posterior distributions.
 
 - Density estimation and plotting tools for analysing and visualising inference results.
 
 - Gaussian-process regression and optimisation.


| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
| [Gibbs Sampling](https://github.com/C-bowman/inference-tools/blob/master/demos/gibbs_sampling_demo.ipynb) <img width="1604" alt="1" src="https://raw.githubusercontent.com/C-bowman/inference-tools/master/docs/source/images/gallery_images/gallery_gibbs_sampling.png"> | [Hamiltonian Monte-Carlo](https://github.com/C-bowman/inference-tools/blob/master/demos/hamiltonian_mcmc_demo.ipynb) <img width="1604" alt="2" src="https://raw.githubusercontent.com/C-bowman/inference-tools/master/docs/source/images/gallery_images/gallery_hmc.png"> | [Density estimation](https://github.com/C-bowman/inference-tools/blob/master/demos/density_estimation_demo.ipynb) <img width="1604" alt="3" src="https://raw.githubusercontent.com/C-bowman/inference-tools/master/docs/source/images/gallery_images/gallery_density_estimation.png"> |
| Matrix plotting <img width="1604" alt="4" src="https://raw.githubusercontent.com/C-bowman/inference-tools/master/docs/source/images/getting_started_images/matrix_plot_example.png"> | Highest-density intervals <img width="1604" alt="5" src="https://raw.githubusercontent.com/C-bowman/inference-tools/master/docs/source/images/gallery_images/gallery_hdi.png"> | [GP regression](https://github.com/C-bowman/inference-tools/blob/master/demos/gp_regression_demo.ipynb) <img width="1604" alt="6" src="https://raw.githubusercontent.com/C-bowman/inference-tools/master/docs/source/images/gallery_images/gallery_gpr.png"> |

## Installation

inference-tools is available from [PyPI](https://pypi.org/project/inference-tools/), 
so can be easily installed using [pip](https://pip.pypa.io/en/stable/) as follows:
```bash
pip install inference-tools
```

## Documentation

Full documentation is available at [inference-tools.readthedocs.io](https://inference-tools.readthedocs.io/en/stable/).