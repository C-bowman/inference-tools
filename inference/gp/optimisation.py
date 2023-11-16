from numpy import array, arange, ndarray, append, maximum
from scipy.optimize import differential_evolution, fmin_l_bfgs_b
from multiprocessing import Pool
from inspect import isclass
from collections.abc import Sequence
import matplotlib.pyplot as plt

from inference.gp.regression import GpRegressor
from inference.gp.covariance import CovarianceFunction, SquaredExponential
from inference.gp.acquisition import ExpectedImprovement
from inference.gp.mean import MeanFunction, ConstantMean


class GpOptimiser:
    """
    A class for performing Gaussian-process optimisation in one or more dimensions.

    GpOptimiser extends the functionality of GpRegressor to perform Gaussian-process
    optimisation, often also referred to as 'Bayesian optimisation'. This technique
    is suited to problems for which a single evaluation of the function being explored
    is expensive, such that the total number of function evaluations must be made as
    small as possible.

    In order to construct the Gaussian-process regression estimate which is used to
    search for the global maximum, on initialisation GpOptimiser must be provided with
    at least two evaluations of the function which is to be maximised.

    :param x: \
        The x-data points as a 2D ``numpy.ndarray`` with shape (number of points,
        number of dimensions). Alternatively, a list of array-like objects can be
        given, which will be converted to a ``ndarray`` internally.

    :param y: \
        The y-data values as a 1D ``numpy.ndarray``.

    :param bounds: \
        An iterable containing tuples which specify the upper and lower bounds
        for the optimisation in each dimension in the format (lower_bound, upper_bound).

    :param y_err: \
        The error on the y-data values supplied as a 1D array.
        This technique explicitly assumes that errors are Gaussian, so the supplied
        error values represent normal distribution standard deviations. If this
        argument is not specified the errors are taken to be small but non-zero.

    :param hyperpars: \
        An array specifying the hyper-parameter values to be used by the
        covariance function class, which by default is ``SquaredExponential``.
        See the documentation for the relevant covariance function class for
        a description of the required hyper-parameters. Generally this argument
        should be left unspecified, in which case the hyper-parameters will be
        selected automatically.

    :param class kernel: \
        The covariance-function class which will be used to model the data. The
        covariance-function classes can be imported from the ``gp`` module and
        then passed to ``GpOptimiser`` using this keyword argument.

    :param class mean: \
        The mean-function class which will be used to model the data. The
        mean-function classes can be imported from the ``gp`` module and
        then passed to ``GpOptimiser`` using this keyword argument.

    :param bool cross_val: \
        If set to ``True``, leave-one-out cross-validation is used to select the
        hyper-parameters in place of the marginal likelihood.

    :param class acquisition: \
        The acquisition-function class which is used to select new points at which
        the objective function is evaluated. The acquisition-function classes can be
        imported from the ``gp`` module and then passed as arguments - see their
        documentation for the list of available acquisition functions. If left unspecified,
        the ``ExpectedImprovement`` acquisition function is used by default.

    :param str optimizer: \
        Selects the optimisation method used for selecting hyper-parameter values and proposed
        evaluations. Available options are "bfgs" for ``scipy.optimize.fmin_l_bfgs_b`` or
        "diffev" for ``scipy.optimize.differential_evolution``.

    :param int n_processes: \
        Sets the number of processes used when selecting hyper-parameters or proposed evaluations.
        Multiple processes are only used when the optimizer keyword is set to "bfgs".
    """

    def __init__(
        self,
        x: ndarray,
        y: ndarray,
        bounds: Sequence,
        y_err: ndarray = None,
        hyperpars: ndarray = None,
        kernel: CovarianceFunction = SquaredExponential,
        mean: MeanFunction = ConstantMean,
        cross_val: bool = False,
        acquisition=ExpectedImprovement,
        optimizer: str = "bfgs",
        n_processes: int = 1,
    ):
        self.x = x if isinstance(x, ndarray) else array(x)
        if self.x.ndim == 1:
            self.x.resize([self.x.size, 1])
        self.y = y if isinstance(y, ndarray) else array(y)
        self.y_err = y_err if isinstance(y_err, (ndarray, type(None))) else array(y_err)

        self.bounds = bounds
        self.kernel = kernel
        self.mean = mean
        self.cross_val = cross_val
        self.n_processes = n_processes
        self.optimizer = optimizer

        self.gp = GpRegressor(
            x=x,
            y=y,
            y_err=y_err,
            hyperpars=hyperpars,
            kernel=kernel,
            mean=mean,
            cross_val=cross_val,
            optimizer=self.optimizer,
            n_processes=self.n_processes,
        )

        # if the class has been passed instead of an instance, create an instance
        self.acquisition = acquisition() if isclass(acquisition) else acquisition
        self.acquisition.update_gp(self.gp)

        # create storage for tracking
        self.acquisition_max_history = []
        self.convergence_metric_history = []
        self.iteration_history = []

    def __call__(self, x):
        return self.gp(x)

    def add_evaluation(self, new_x: ndarray, new_y: ndarray, new_y_err: ndarray = None):
        """
        Add the latest evaluation to the data set and re-build the
        Gaussian process so a new proposed evaluation can be made.

        :param new_x: location of the new evaluation
        :param new_y: function value of the new evaluation
        :param new_y_err: Error of the new evaluation.
        """
        new_x = new_x if isinstance(new_x, ndarray) else array(new_x)
        if new_x.shape != (1, self.x.shape[1]):
            new_x.resize((1, self.x.shape[1]))
        new_y = new_y if isinstance(new_y, ndarray) else array(new_y)
        good_type = isinstance(new_y_err, (ndarray, type(None)))
        new_y_err = new_y_err if good_type else array(new_y_err)

        # store the acquisition function value of the new point
        self.acquisition_max_history.append(self.acquisition(new_x))
        self.convergence_metric_history.append(
            self.acquisition.convergence_metric(new_x)
        )
        self.iteration_history.append(self.y.size + 1)

        # update the data arrays
        self.x = append(self.x, new_x, axis=0)
        self.y = append(self.y, new_y)

        if self.y_err is not None:
            if new_y_err is not None:
                self.y_err = append(self.y_err, new_y_err)
            else:
                raise ValueError(
                    "y_err must be specified for new evaluations if y_err was specified during __init__"
                )

        # re-train the GP
        self.gp = GpRegressor(
            x=self.x,
            y=self.y,
            y_err=self.y_err,
            kernel=self.kernel,
            mean=self.mean,
            cross_val=self.cross_val,
            optimizer=self.optimizer,
            n_processes=self.n_processes,
        )
        self.mu_max = self.y.max()

        # update the acquisition function info
        self.acquisition.update_gp(self.gp)

    def diff_evo(self):
        opt_result = differential_evolution(
            self.acquisition.opt_func, self.bounds, popsize=30
        )
        solution = opt_result.x
        funcval = opt_result.fun
        if hasattr(funcval, "__len__"):
            funcval = funcval[0]
        return solution, funcval

    def launch_bfgs(self, x0: ndarray):
        return fmin_l_bfgs_b(
            self.acquisition.opt_func_gradient,
            x0,
            approx_grad=False,
            bounds=self.bounds,
            pgtol=1e-10,
        )

    def multistart_bfgs(self):
        starting_positions = self.acquisition.starting_positions(self.bounds)
        # run BFGS for each starting position
        if self.n_processes == 1:
            results = [self.launch_bfgs(x0) for x0 in starting_positions]
        else:
            workers = Pool(self.n_processes)
            results = workers.map(self.launch_bfgs, starting_positions)
        # extract best solution
        best_result = sorted(results, key=lambda x: float(x[1]))[0]
        solution = best_result[0]
        funcval = float(best_result[1])
        return solution, funcval

    def propose_evaluation(self, optimizer=None):
        """
        Request a proposed location for the next evaluation. This proposal is
        selected by maximising the chosen acquisition function.

        :param str optimizer: \
            Selects the optimization method used for selecting the proposed evaluation.
            Available options are "bfgs" for ``scipy.optimize.fmin_l_bfgs_b`` or
            "diffev" for ``scipy.optimize.differential_evolution``. This keyword allows
            the user to override the choice of optimizer given when ``GpOptimiser`` was
            initialised.

        :return: location of the next proposed evaluation.
        """
        opt = optimizer if optimizer is not None else self.optimizer
        if opt == "bfgs":
            # find the evaluation point which maximises the acquisition function
            proposed_ev, max_acq = self.multistart_bfgs()
        else:
            proposed_ev, max_acq = self.diff_evo()
        # if the problem is 1D, but the result is returned as a length-1 array,
        # extract the result from the array
        if hasattr(proposed_ev, "__len__") and len(proposed_ev) == 1:
            proposed_ev = proposed_ev[0]
        return proposed_ev

    def plot_results(self, filename=None, show_plot=True):
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(121)
        maxvals = maximum.accumulate(self.y)
        pad = maxvals.ptp() * 0.1
        iterations = arange(len(self.y)) + 1
        ax1.plot(iterations, maxvals, c="red", alpha=0.6, label="max observed value")
        ax1.plot(iterations, self.y, ".", label="function evaluations", markersize=10)
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("function value")
        ax1.set_ylim([maxvals.min() - pad, maxvals.max() + pad])
        ax1.legend(loc=4)
        ax1.grid()

        ax2 = fig.add_subplot(122)
        ax2.plot(
            self.iteration_history, self.convergence_metric_history, c="C0", alpha=0.35
        )
        ax2.plot(
            self.iteration_history,
            self.convergence_metric_history,
            ".",
            c="C0",
            label=self.acquisition.convergence_description,
            markersize=10,
        )
        ax2.set_yscale("log")
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("acquisition function value")
        ax2.set_xlim([0, None])
        ax2.set_title("Convergence summary")
        ax2.legend()
        ax2.grid()

        fig.tight_layout()

        if filename is not None:
            plt.savefig(filename)
        if show_plot:
            plt.show()
        else:
            plt.close()
