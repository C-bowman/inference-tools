from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from numpy import ndarray, linspace, sqrt
from scipy.integrate import quad


class DensityEstimator(ABC):
    """
    Abstract base class for 1D density estimators.
    """

    @abstractmethod
    def __call__(self, x: ndarray) -> ndarray:
        pass

    @abstractmethod
    def cdf(self, x: ndarray) -> ndarray:
        pass

    @abstractmethod
    def moments(self) -> tuple:
        pass

    def interval(self, frac=0.95):
        p_max = self(self.mode)
        p_conf = self.binary_search(
            self.interval_prob, frac, [0.0, p_max], uphill=False
        )
        return self.get_interval(p_conf)

    def get_interval(self, z):
        lwr = self.binary_search(self, z, [self.lwr_limit, self.mode], uphill=True)
        upr = self.binary_search(self, z, [self.mode, self.upr_limit], uphill=False)
        return lwr, upr

    def interval_prob(self, z):
        lwr, upr = self.get_interval(z)
        return quad(self, lwr, upr, limit=100)[0]

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

        def ensure_is_nested_list(var):
            if not isinstance(var[0], (list, tuple)):
                var = [var]
            return var

        sigma_1 = ensure_is_nested_list(self.interval(frac=0.68268))
        sigma_2 = ensure_is_nested_list(self.interval(frac=0.95449))
        sigma_3 = ensure_is_nested_list(self.interval(frac=0.9973))
        mu, var, skw, kur = self.moments()

        s_min = sigma_3[0][0]
        s_max = sigma_3[-1][1]

        lwr = s_min - 0.1 * (s_max - s_min)
        upr = s_max + 0.1 * (s_max - s_min)

        axis = linspace(lwr, upr, 500)

        fig, ax = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(10, 6),
            gridspec_kw={"width_ratios": [2, 1]},
        )
        ax[0].plot(axis, self(axis), lw=1, c="C0")
        ax[0].fill_between(axis, self(axis), color="C0", alpha=0.1)
        ax[0].plot([self.mode, self.mode], [0.0, self(self.mode)], c="red", ls="dashed")

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
            for itvl in sigma:
                ax[1].text(
                    x2,
                    height,
                    rf"{itvl[0]:.5G} $\rightarrow$ {itvl[1]:.5G}",
                    horizontalalignment="left",
                )
                height -= gap
            return height

        h = write_sigma(h, "1-sigma:", sigma_1)
        h = write_sigma(h, "2-sigma:", sigma_2)
        h = write_sigma(h, "3-sigma:", sigma_3)
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

    @staticmethod
    def binary_search(func, value, bounds, uphill=True):
        x_min, x_max = bounds
        x = (x_min + x_max) * 0.5

        converged = False
        while not converged:
            f = func(x)
            if f > value:
                if uphill:
                    x_max = x
                else:
                    x_min = x
            else:
                if uphill:
                    x_min = x
                else:
                    x_max = x

            x = (x_min + x_max) * 0.5
            if abs((x_max - x_min) / x) < 1e-3:
                converged = True

        # now linearly interpolate as a polish step
        f_max = func(x_max)
        f_min = func(x_min)
        df = f_max - f_min

        return x_min * ((f_max - value) / df) + x_max * ((value - f_min) / df)
