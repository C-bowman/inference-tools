from numpy import linspace
from numpy.random import normal, random, exponential
from inference.pdf import UnimodalPdf
import matplotlib.pyplot as plt

N = 5000
s1 = normal(size=N) * 0.5 + exponential(size=N)
s2 = normal(size=N) * 0.5 + 3 * random(size=N) + 2.5

pdf_axis = linspace(-2, 7.5, 100)
pdf1 = UnimodalPdf(s1)
pdf2 = UnimodalPdf(s2)


fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)
ax.plot(pdf_axis, pdf1(pdf_axis), alpha=0.75, c="C0", lw=2)
ax.fill_between(
    pdf_axis, pdf1(pdf_axis), alpha=0.2, color="C0", label="exponential + gaussian"
)
ax.plot(pdf_axis, pdf2(pdf_axis), alpha=0.75, c="C1", lw=2)
ax.fill_between(
    pdf_axis, pdf2(pdf_axis), alpha=0.2, color="C1", label="uniform + gaussian"
)
ax.set_ylim([0.0, None])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlabel("parameter value")
ax.set_ylabel("probability density")
ax.grid()
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig("gallery_density_estimation.png")
plt.show()
