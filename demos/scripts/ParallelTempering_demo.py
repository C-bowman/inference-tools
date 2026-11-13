from numpy import log, sqrt, sin, arctan2, pi

# define a posterior with multiple separate peaks
def multimodal_posterior(theta):
    x, y = theta
    r = sqrt(x**2 + y**2)
    phi = arctan2(y, x)
    z = (r - (0.5 + pi - phi*0.5)) / 0.1
    return -0.5*z**2 + 4*log(sin(phi*2.)**2)


# required for multi-process code when running on windows
if __name__ == "__main__":

    from inference.mcmc import GibbsChain, ParallelTempering

    # define a set of temperature levels
    N_levels = 6
    temps = [10**(2.5*k/(N_levels-1.)) for k in range(N_levels)]

    # create a set of chains - one with each temperature
    chains = [GibbsChain(posterior=multimodal_posterior, start=[0.5, 0.5], temperature=T) for T in temps]

    # Retrieve the chains before the context shuts down the worker processes.
    with ParallelTempering(chains=chains) as PT:
        PT.run_for(minutes=0.5)
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
