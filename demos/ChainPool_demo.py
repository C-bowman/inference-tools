
from inference.mcmc import GibbsChain, ChainPool
from time import time


def rosenbrock(t):
    # This is a modified form of the rosenbrock function, which
    # is commonly used to test optimisation algorithms
    X, Y = t
    X2 = X**2
    b = 15  # correlation strength parameter
    v = 3   # variance of the gaussian term
    return -X2 - b*(Y - X2)**2 - 0.5*(X2 + Y**2)/v


# required for multi-process code when running on windows
if __name__ == "__main__":

    """
    The ChainPool class provides a convenient means to store multiple
    chain objects, and simultaneously advance those chains using multiple
    python processes.
    """

    # for example, here we create a singular chain object
    chain = GibbsChain(posterior = rosenbrock, start = [0.,0.])
    # then advance it for some number of samples, and note the run-time
    t1 = time()
    chain.advance(150000)
    t2 = time()
    print('time elapsed, single chain:', t2-t1)


    # We may want to run a number of chains in parallel - for example multiple chains
    # over different posteriors, or on a single posterior with different starting locations.

    # Here we create two chains with different starting points:
    chain_1 = GibbsChain(posterior = rosenbrock, start = [0.,0.])
    chain_2 = GibbsChain(posterior = rosenbrock, start = [0.,0.])

    # now we pass those chains to ChainPool in a list
    cpool = ChainPool( [chain_1, chain_2] )

    # if we now wish to advance both of these chains some number of steps, and do so in
    # parallel, we can use the advance() method of the ChainPool instance:
    t1 = time()
    cpool.advance(150000)
    t2 = time()
    print('time elapsed, two chains:', t2-t1)

    # assuming you are running this example on a machine with two free cores, advancing
    # both chains in this way should have taken a comparable time to advancing just one.

