from numpy import log, sqrt, sin, arctan2, pi
from inference.mcmc import GibbsChain, ParallelTempering


def multimodal_posterior(theta):
    x, y = theta
    r = sqrt(x**2 + y**2)
    phi = arctan2(y, x)
    z = (r - (0.5 + pi - phi * 0.5)) / 0.1
    return -0.5 * z**2 + 4 * log(sin(phi * 2.0) ** 2)


if __name__ == "__main__":
    temps = ParallelTempering.generate_temperatures(
        n_chains=6, n_dimensions=2
    )
    chains = [
        GibbsChain(posterior=multimodal_posterior, start=[0.5, 0.5], temperature=T)
        for T in temps
    ]

    with ParallelTempering(chains=chains) as PT:
        PT.run_for(minutes=0.5)
        chains = PT.return_chains()

    chains[0].trace_plot(filename="parallel_tempering_trace.png")
    chains[0].matrix_plot(filename="parallel_tempering_matrix.png")
