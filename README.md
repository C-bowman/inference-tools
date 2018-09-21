# The inference_tools repository
This package aims to provide a set of python-based tools for Bayesian data analysis
which are simple to use, allowing them to applied quickly and easily.

The package documentation can be found in `/docs/inference_tools_documentation.pdf`. This documentation contains full
descriptions of all available classes (and their methods), but does not yet contain detailed code examples of how
to use these classes.

For example code, instead see the /demos/ folder which contains a demonstration code for every available class,
including detailed comments.

Requests for features/improvements can be made via the issue tracker. If you have questions
or are interested in getting involved with the development of this package, please contact
me at `chris.bowman@york.ac.uk`.

## modules

### mcmc
This module is designed to provide Markov-Chain Monte-Carlo (MCMC) samplers which can
be easily applied to inference problems.

**GibbsChain**  
At present the primary sampling tool in the module is the GibbsChain class, which performs
Gibbs sampling. This implementation is self-tuning, such that detailed knowledge of the PDF
being sampled is not required in order for the algorithm to work efficiently.

Demonstrations of how to use the GibbsChain class can be found in `/demos/GibbsChain_demo.py` and
`/demos/spectroscopy_demo.py`

### gp_tools
This module provides implementations of some useful applications of 'Gaussian processes'.
This involves modelling data through multivariate normal distributions where the
covariance of any two points is defined by the 'distance' between them in some arbitrary
space.

**GpRegressor**  
Gaussian-process regression has been implemented as the GpRegressor class, and can fit data which is
arbitrarily spaced (i.e. non-gridded) in any number of dimensions. A key advantage this
technique holds over other regression methods is its ability to properly account for
errors on the data and provide a corresponding error on the regression estimate.

A demonstration of how to use the GpRegressor class can be found in `/demos/GpRegressor_demo.py`

**GpInverter**  
A Gaussian-process solver for linear-inverse problems is also available, implemented as
the GpInverter class. This technique is ideal for solving linear inverse problems in
which the model parameters (or 'solution vector') correspond to some physical quantity
of interest at different points in a space, and we expect this quantity to vary smoothly
across the space.

A demonstration of how to use the GpInverter class can be found in `/demos/GpInverter_demo.py`

### pdf_tools
This module provides tools for reconstructing the probability density function from which
a given set of samples was drawn.

**UnimodalPdf**
UnimodalPdf provides smooth estimates of univariate, unimodal PDFs by fitting an extremely flexible
parametric model to a given set of sample data. The class also provides a method which returns the
credible interval for a chosen percentile.

A demonstration of how to use the UnimodalPdf class can be found in `/demos/UnimodalPdf_demo.py`

**GaussianKDE**
The GaussianKDE class provides an estimate of a univariate PDF for a given set of sample data using
Gaussian kernel density estimation. An estimate of the mode is calulated automatically, and like
UnimodalPdf, the credible interval for a given percentile is available through a method call.

A demonstration of how to use the UnimodalPdf class can be found in `/demos/GaussianKDE_demo.py`