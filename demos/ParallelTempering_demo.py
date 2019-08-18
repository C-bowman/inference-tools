
from numpy import log, sqrt, sin, arctan2, pi

# define a posterior with multiple separate peaks
def multimodal_posterior(theta):
    x,y = theta
    r = sqrt(x**2 + y**2)
    phi = arctan2(y,x)
    z = ((r - (0.5 + pi - phi*0.5))/0.1)
    return -0.5*z**2  + 4*log(sin(phi*2.)**2)

from inference.mcmc import GibbsChain, ParallelTempering

# define a set of temperature levels
N_levels = 6
temps = [10**(2.5*k/(N_levels-1.)) for k in range(N_levels)]

# create a set of chains - one with each temperature
chains = [ GibbsChain( posterior=multimodal_posterior, start = [0.5,0.5], temperature=T) for T in temps ]

PT = ParallelTempering(chains=chains)

for i in range(5000):
    PT.advance(10)
    PT.swap()

 
chains = PT.return_chains()

chains[0].plot_diagnostics()
chains[0].trace_plot()
chains[0].matrix_plot()

# trigger the shutdown event to terminate all the processes
PT.shutdown()