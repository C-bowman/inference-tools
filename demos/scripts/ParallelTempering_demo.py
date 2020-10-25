
from numpy import log, sqrt, sin, arctan2, pi

# define a posterior with multiple separate peaks
def multimodal_posterior(theta):
    x,y = theta
    r = sqrt(x**2 + y**2)
    phi = arctan2(y,x)
    z = ((r - (0.5 + pi - phi*0.5))/0.1)
    return -0.5*z**2  + 4*log(sin(phi*2.)**2)


# required for multi-process code when running on windows
if __name__ == "__main__":

    from inference.mcmc import GibbsChain, ParallelTempering

    # define a set of temperature levels
    N_levels = 6
    temps = [10**(2.5*k/(N_levels-1.)) for k in range(N_levels)]

    # create a set of chains - one with each temperature
    chains = [ GibbsChain( posterior=multimodal_posterior, start = [0.5,0.5], temperature=T) for T in temps ]

    # When an instance of ParallelTempering is created, a dedicated process for each chain is spawned.
    # These separate processes will automatically make use of the available cpu cores, such that the
    # computations to advance the separate chains are performed in parallel.
    PT = ParallelTempering(chains=chains)

    # These processes wait for instructions which can be sent using the methods of the
    # ParallelTempering object:
    PT.run_for(minutes=0.5)

    # To recover a copy of the chains held by the processes
    # we can use the return_chains method:
    chains = PT.return_chains()

    # by looking at the trace plot for the T = 1 chain, we see that it makes
    # large jumps across the parameter space due to the swaps.
    chains[0].trace_plot()

    # Even though the posterior has strongly separated peaks, the T = 1 chain
    # was able to explore all of them due to the swaps.
    chains[0].matrix_plot()

    # We can also visualise the acceptance rates of proposed position swaps between
    # each chain using the swap_diagnostics method:
    PT.swap_diagnostics()

    # Because each process waits for instructions from the ParallelTempering object,
    # they will not self-terminate. To terminate all the processes we have to trigger
    # a shutdown even using the shutdown method:
    PT.shutdown()