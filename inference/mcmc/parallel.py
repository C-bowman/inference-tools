import sys
from warnings import warn
from multiprocessing import Process, Pipe, Event, Pool
from multiprocessing.connection import Connection
from time import time
from random import choice

import matplotlib.pyplot as plt
from numpy import arange, exp, identity, zeros
from numpy.random import random, shuffle, seed, randint
from inference.plotting import transition_matrix_plot
from inference.mcmc.base import MarkovChain


class ChainPool:
    def __init__(self, chains: list[MarkovChain]):
        self.chains = chains
        self.pool_size = len(self.chains)
        self.pool = Pool(self.pool_size)

    def advance(self, n):
        self.chains = self.pool.map(
            self.adv_func, [(n, chain) for chain in self.chains]
        )

    @staticmethod
    def adv_func(arg):
        n, chain = arg
        chain.advance(n)
        return chain


def tempering_process(
    chain: MarkovChain, connection: Connection, end: Event, proc_seed: int
):
    # used to ensure each process has a different random seed
    seed(proc_seed)
    # main loop
    while not end.is_set():
        # poll the pipe until there is something to read
        while not end.is_set():
            if connection.poll(timeout=0.05):
                D = connection.recv()
                break

        # if read loop was broken because of shutdown event
        # then break the main loop as well
        if end.is_set():
            break

        task = D["task"]

        # advance the chain
        if task == "advance":
            for _ in range(D["advance_count"]):
                chain.take_step()
            connection.send("advance_complete")  # send signal to confirm completion

        # return the current position of the chain
        elif task == "send_position":
            connection.send((chain.get_last(), chain.probs[-1]))

        # update the position of the chain
        elif task == "update_position":
            chain.replace_last(D["position"])
            chain.probs[-1] = D["probability"] * chain.inv_temp

        # return the local chain object
        elif task == "send_chain":
            connection.send(chain)


class ParallelTempering:
    """
    A class which enables 'parallel tempering', a sampling algorithm which
    advances multiple Markov-chains in parallel, each with a different
    'temperature', with a probability that the chains will exchange their
    positions during the advancement.

    The 'temperature' concept introduces a transformation to the distribution
    being sampled, such that a chain with temperature 'T' instead samples from
    the provided posterior distribution raised to the power 1/T.

    When T = 1, the original distribution is recovered, but choosing T > 1 has
    the effect of 'compressing' the distribution, such that any two points having
    different probability densities will have the difference between those densities
    reduced as the temperature is increased. This allows chains with higher
    temperatures to take much larger steps, and explore the distribution more
    quickly.

    Parallel tempering exploits this by advancing a collection of markov-chains at
    different temperatures, with at least one chain at T = 1 (i.e. sampling from
    the actual posterior distribution). At regular intervals, pairs of chains are
    selected at random and a metropolis-hastings test is performed to decide if
    the pair exchange their positions.

    The ability for the T = 1 chain to exchange positions with chains of higher
    temperatures allows it to make large jumps to other areas of the distribution
    which it may take a large number of steps to reach otherwise.

    This is particularly useful when sampling from highly-complex distributions
    which may have many separate maxima and/or strong correlations.

    :param chains: \
        A list of Markov-Chain objects (such as GibbsChain, PcaChain, HamiltonianChain)
        covering a range of different temperature levels. The list of chains should be
        sorted in order of increasing chain temperature.
    """

    def __init__(self, chains: list[MarkovChain]):
        self.shutdown_evt = Event()
        self.connections = []
        self.processes = []
        self.temperatures = [1.0 / chain.inv_temp for chain in chains]
        self.inv_temps = [chain.inv_temp for chain in chains]
        self.N_chains = len(chains)

        self.attempted_swaps = identity(self.N_chains)
        self.successful_swaps = zeros([self.N_chains, self.N_chains])

        if sorted(self.temperatures) != self.temperatures:
            warn(
                """
                The list of Markov-chain objects passed to ParallelTempering
                should be sorted in order of increasing chain temperature.
                """
            )

        # Spawn a separate process for each chain object
        for chn in chains:
            parent_ctn, child_ctn = Pipe()
            self.connections.append(parent_ctn)
            p = Process(
                target=tempering_process,
                args=(chn, child_ctn, self.shutdown_evt, randint(30000)),
            )
            self.processes.append(p)

        [p.start() for p in self.processes]

    def take_steps(self, n: int):
        """
        Advance all the chains *n* steps without performing any swaps.

        :param int n: The number of steps by which every chain is advanced.
        """
        # order the chains to advance n steps
        D = {"task": "advance", "advance_count": n}
        for pipe in self.connections:
            pipe.send(D)

        # block until all chains report successful advancement
        responses = [pipe.recv() == "advance_complete" for pipe in self.connections]
        if not all(responses):
            raise ValueError("Unexpected data received from pipe")

    def uniform_pairs(self):
        """
        Randomly pair up each chain, with uniform sampling across all possible pairings
        """
        proposed_swaps = arange(self.N_chains)
        shuffle(proposed_swaps)
        return [p for p in zip(proposed_swaps[::2], proposed_swaps[1::2])]

    def tight_pairs(self):
        """
        Randomly pair up each chain, with almost all paired chains being separated
        by either 1 or 2 temperature levels.
        """
        # first generate all possible pairings with a gap of 2 or less
        pairs = [(i, i + j) for i in range(self.N_chains - 1) for j in [1, 2]][:-1]
        sample = []
        # randomly sample from these pairings until no valid pairs remain
        while len(pairs) > 0:
            p = choice(pairs)
            pairs = [k for k in pairs if not any(j in k for j in p)]
            sample.append(p)
        # if there are still some pairs which haven't been paired, randomly pair the remaining ones
        remaining = len(sample) - self.N_chains // 2
        if remaining != 0:
            leftovers = [
                i for i in range(self.N_chains) if not any(i in p for p in sample)
            ]
            shuffle(leftovers)
            sample.extend(
                [
                    p if p[0] < p[1] else (p[1], p[0])
                    for p in zip(leftovers[::2], leftovers[1::2])
                ]
            )
        return sample

    def swap(self):
        """
        Randomly group all chains into pairs and propose a position swap between each pair.
        """
        # ask each process to report the current position of its chain
        D = {"task": "send_position"}
        [pipe.send(D) for pipe in self.connections]

        # receive the positions and probabilities
        data = [pipe.recv() for pipe in self.connections]
        positions = [k[0] for k in data]
        probabilities = [k[1] for k in data]

        # randomly pair up indices for all the processes
        proposed_swaps = self.tight_pairs()

        # perform MH tests to see if the swaps occur or not
        for pair in proposed_swaps:
            self.attempted_swaps[pair] += 1

        for i, j in proposed_swaps:
            dt = self.inv_temps[i] - self.inv_temps[j]
            pi = probabilities[i] / self.inv_temps[i]
            pj = probabilities[j] / self.inv_temps[j]
            dp = pi - pj

            if random() <= exp(-dt * dp):  # check if the swap is successful
                Di = {
                    "task": "update_position",
                    "position": positions[i],
                    "probability": pi,
                }

                Dj = {
                    "task": "update_position",
                    "position": positions[j],
                    "probability": pj,
                }

                self.connections[i].send(Dj)
                self.connections[j].send(Di)
                self.successful_swaps[i, j] += 1

    def advance(self, n, swap_interval=10):
        """
        Advances each chain by a total of *n* steps, performing swap attempts
        at intervals set by the *swap_interval* keyword.

        :param int n: The number of steps each chain will advance.
        :param int swap_interval: \
            The number of steps that are taken in each chain between swap attempts.
        """
        k = 50  # divide chain steps into k groups to track progress
        total_cycles = n // swap_interval
        if k < total_cycles:
            k = total_cycles
            cycles = 1
        else:
            cycles = total_cycles // k

        t_start = time()
        for j in range(k):
            for i in range(cycles):
                self.take_steps(swap_interval)
                self.swap()

            dt = time() - t_start

            # display the progress status message
            pct = str(int(100 * (j + 1) / k))
            eta = str(int(dt * (k / (j + 1) - 1)))
            sys.stdout.write(
                f"\r  [ Running ParallelTempering - {pct}% complete   ETA: {eta} sec ]    "
            )
            sys.stdout.flush()

        # run the remaining cycles
        if total_cycles % k != 0:
            for i in range(total_cycles % k):
                self.take_steps(swap_interval)
                self.swap()

        # run remaining steps
        if n % swap_interval != 0:
            self.take_steps(n % swap_interval)

        # print the completion message
        sys.stdout.write(
            "\r  [ Running ParallelTempering - complete! ]                    "
        )
        sys.stdout.flush()
        sys.stdout.write("\n")

    def run_for(self, minutes=0, hours=0, swap_interval=10):
        """
        Advances all chains for a chosen amount of computation time.

        :param float minutes: Number of minutes for which to advance the chains.
        :param float hours: Number of hours for which to advance the chains.
        :param int swap_interval: \
            The number of steps that are taken in each chain between swap attempts.
        """
        # first find the runtime in seconds:
        run_time = (hours * 60.0 + minutes) * 60.0
        start_time = time()
        end_time = start_time + run_time

        # estimate how long it takes to do one swap cycle
        t1 = time()
        self.take_steps(swap_interval)
        self.swap()
        t2 = time()

        # number of cycles chosen to give a print-out roughly every 2 seconds
        N = max(1, int(2.0 / (t2 - t1)))

        while time() < end_time:
            for i in range(N):
                self.take_steps(swap_interval)
                self.swap()

            # display the progress status message
            seconds_remaining = end_time - time()
            m, s = divmod(seconds_remaining, 60)
            h, m = divmod(m, 60)
            time_left = "%d:%02d:%02d" % (h, m, s)
            sys.stdout.write(
                f"\r  [ Running ParallelTempering - time remaining: {time_left} ]    "
            )
            sys.stdout.flush()

        # this is a little ugly...
        sys.stdout.write(
            "\r  [ Running ParallelTempering - complete! ]                    "
        )
        sys.stdout.flush()
        sys.stdout.write("\n")

    def swap_diagnostics(self):
        """
        Plot the acceptance rates of proposed position swaps between the
        different chains. This can be useful in selecting appropriate temperatures
        for the chains.
        """
        rate_matrix = self.successful_swaps / self.attempted_swaps.clip(min=1)

        pairs = [
            (i, i + j)
            for j in range(1, self.N_chains)
            for i in range(self.N_chains - j)
        ]
        total_swaps = zeros(self.N_chains)
        for i, j in pairs:
            total_swaps[i] += self.successful_swaps[i, j]
            total_swaps[j] += self.successful_swaps[i, j]

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        transition_matrix_plot(
            ax=ax1, matrix=rate_matrix, exclude_diagonal=True, upper_triangular=True
        )
        ax1.set_xlabel("chain number")
        ax1.set_ylabel("chain number")
        ax1.set_title("acceptance rate of chain position swaps")

        ax2 = fig.add_subplot(122)
        ax2.bar([i for i in range(1, self.N_chains + 1)], total_swaps)
        ax2.set_ylim([0, None])
        ax2.set_xlabel("chain number")
        ax2.set_ylabel("total successful position swaps")

        plt.tight_layout()
        plt.show()

    def return_chains(self) -> list[MarkovChain]:
        """
        Recover the chain held by each process and return them in a list.

        :return: A list containing the chain objects.
        """
        # order each process to return its locally stored chain object
        request = {"task": "send_chain"}
        for pipe in self.connections:
            pipe.send(request)

        # receive the chains and return them
        return [pipe.recv() for pipe in self.connections]

    def shutdown(self):
        """
        Trigger a shutdown event which tells the processes holding each of
        the chains to terminate.
        """
        self.shutdown_evt.set()
        [p.join() for p in self.processes]
