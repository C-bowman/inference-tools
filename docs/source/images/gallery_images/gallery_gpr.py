import matplotlib.pyplot as plt
from numpy import linspace, array
from inference.gp import GpRegressor, SquaredExponential


# initialise the class with the data and errors
x_fit = linspace(0, 5, 200)
x = array([0.5, 1.0, 1.5, 3.0, 3.5, 4.0, 4.5])
y = array([0.157, -0.150, -0.305, -0.049, 0.366, 0.417, 0.430]) * 10.0
y_errors = array([0.1, 0.01, 0.1, 0.4, 0.1, 0.01, 0.1]) * 10.0
gpr = GpRegressor(x, y, y_err=y_errors, kernel=SquaredExponential())
mu, sig = gpr(x_fit)

# now plot the regression estimate and the data together
col = "blue"
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)
ax.fill_between(
    x_fit, mu - sig, mu + sig, color=col, alpha=0.2, label="GPR uncertainty"
)
ax.fill_between(x_fit, mu - 2 * sig, mu + 2 * sig, color=col, alpha=0.1)
ax.plot(x_fit, mu, lw=2, c=col, label="GPR mean")
ax.errorbar(
    x,
    y,
    yerr=y_errors,
    marker="o",
    color="black",
    ecolor="black",
    ls="none",
    label="data values",
    markerfacecolor="none",
    markeredgewidth=2,
    markersize=10,
    elinewidth=2
)
ax.set_xlim([0, 5])
ax.set_ylim([-7, 7])
ax.set_xlabel("x-data value", fontsize=11)
ax.set_ylabel("y-data value", fontsize=11)
ax.grid()
ax.legend(loc=4, fontsize=12)
plt.tight_layout()
plt.savefig("gallery_gpr.png")
plt.show()
