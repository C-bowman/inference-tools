from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from numpy import array, ndarray, linspace, sqrt
from scipy.optimize import minimize
from inference.pdf.hdi import sample_hdi


class DensityEstimator(ABC):
    """
    Abstract base class for 1D density estimators.
    """

    sample: ndarray
    mode: float

    @abstractmethod
    def __call__(self, x: ndarray) -> ndarray:
        pass

    @abstractmethod
    def cdf(self, x: ndarray) -> ndarray:
        pass

    @abstractmethod
    def moments(self) -> tuple:
        pass

    def interval(self, fraction: float) -> tuple[float, float]:
        """
        Calculates the 'highest-density interval', the shortest single interval
        which contains a chosen fraction of the total probability.

        :param fraction: \
            Fraction of the total probability contained by the interval. The given
            value must be between 0 and 1.

        :return: \
            A tuple of the lower and upper limits of the highest-density interval
            in the form ``(lower_limit, upper_limit)``.
        """
        if not 0.0 < fraction < 1.0:
            raise ValueError(
                f"""\n
                \r[ {self.__class__.__name__} error ]
                \r>> The 'fraction' argument must have a value greater than
                \r>> zero and less than one, but the value given was {fraction}.
                """
            )
        # use the sample to estimate the HDI
        lwr, upr = sample_hdi(self.sample, fraction=fraction)
        # switch variables to the centre and width of the interval
        c = 0.5 * (lwr + upr)
        w = upr - lwr

        simplex = array([[c, w], [c, 0.95 * w], [c - 0.05 * w, w]])
        weight = 0.2 / self(self.mode)
        result = minimize(
            fun=self.__hdi_cost,
            x0=simplex[0, :],
            method="Nelder-Mead",
            options={"initial_simplex": simplex},
            args=(fraction, weight),
        )
        c, w = result.x
        return c - 0.5 * w, c + 0.5 * w

    def __hdi_cost(self, theta, fraction, prob_weight):
        c, w = theta
        v = array([c - 0.5 * w, c + 0.5 * w])
        Pa, Pb = self(v)
        Fa, Fb = self.cdf(v)
        return (prob_weight * (Pa - Pb)) ** 2 + (Fb - Fa - fraction) ** 2

    def plot_summary(self, filename=None, show=True, label=None):
        """
        Plot the estimated PDF along with summary statistics.

        :keyword str filename: \
            Filename to which the plot will be saved. If unspecified, the plot will not be saved.

        :keyword bool show: \
            Boolean value indicating whether the plot should be displayed in a window. (Default is True)

        :keyword str label: \
            The label to be used for the x-axis on the plot as a string.
        """

        sigma_1 = self.interval(fraction=0.68268)
        sigma_2 = self.interval(fraction=0.95449)
        mu, var, skw, kur = self.moments()
        s_min, s_max = sigma_2
        maxprob = self(self.mode)

        delta = 0.1 * (s_max - s_min)
        lwr = s_min - delta
        upr = s_max + delta
        while self(lwr) / maxprob > 5e-3:
            lwr -= delta
        while self(upr) / maxprob > 5e-3:
            upr += delta

        axis = linspace(lwr, upr, 500)

        fig, ax = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(10, 6),
            gridspec_kw={"width_ratios": [2, 1]},
        )
        ax[0].plot(axis, self(axis), lw=1, c="C0")
        ax[0].fill_between(axis, self(axis), color="C0", alpha=0.1)
        ax[0].plot([self.mode, self.mode], [0.0, maxprob], c="red", ls="dashed")

        ax[0].set_xlabel(label or "argument", fontsize=13)
        ax[0].set_ylabel("probability density", fontsize=13)
        ax[0].set_ylim([0.0, None])
        ax[0].grid()

        gap = 0.05
        h = 0.95
        x1 = 0.35
        x2 = 0.40

        def section_title(height, name):
            ax[1].text(0.0, height, name, horizontalalignment="left", fontweight="bold")
            return height - gap

        def write_quantity(height, name, value):
            ax[1].text(x1, height, f"{name}:", horizontalalignment="right")
            ax[1].text(x2, height, f"{value:.5G}", horizontalalignment="left")
            return height - gap

        h = section_title(h, "Basics")
        h = write_quantity(h, "Mode", self.mode)
        h = write_quantity(h, "Mean", mu)
        h = write_quantity(h, "Standard dev", sqrt(var))
        h -= gap

        h = section_title(h, "Highest-density intervals")

        def write_sigma(height, name, sigma):
            ax[1].text(x1, height, name, horizontalalignment="right")
            ax[1].text(
                x2,
                height,
                rf"{sigma[0]:.5G} $\rightarrow$ {sigma[1]:.5G}",
                horizontalalignment="left",
            )
            height -= gap
            return height

        h = write_sigma(h, "1-sigma:", sigma_1)
        h = write_sigma(h, "2-sigma:", sigma_2)
        h -= gap

        h = section_title(h, "Higher moments")
        h = write_quantity(h, "Variance", var)
        h = write_quantity(h, "Skewness", skw)
        h = write_quantity(h, "Kurtosis", kur)

        ax[1].axis("off")

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()

        return fig, ax
