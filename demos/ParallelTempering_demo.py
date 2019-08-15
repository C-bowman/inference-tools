
from numpy import log, sqrt, sin, arctan2

def multimodal_posterior(theta):
    x,y = theta
    r = sqrt(x**2 + y**2)
    phi = arctan2(y,x)
    return -0.5*((r - 1.)/0.1)**2  + 2*log(sin(phi*1.5)**2)

from inference.mcmc import GibbsChain, ParallelTempering

temps = [2.**k for k in range(6)]
chains = [ GibbsChain( posterior=multimodal_posterior, start = [0.5,0.5], temperature=T) for T in temps ]
for c in chains:
    c.print_status = False


PT = ParallelTempering(chains=chains)

for i in range(500):
    PT.advance(50)
    PT.swap()

chains = PT.return_chains()

chains[0].plot_diagnostics()

chains[0].trace_plot()

chains[0].matrix_plot()

PT.shutdown()