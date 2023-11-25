from abc import ABC, abstractmethod
from copy import copy
from time import time
from numpy import ndarray
from inference.mcmc.utilities import ChainProgressPrinter


class MarkovChain(ABC):
    chain_length: int
    ProgressPrinter: ChainProgressPrinter

    def advance(self, m: int):
        """
        Advances the chain by taking ``m`` new steps.

        :param int m: Number of steps the chain will advance.
        """
        k = 100  # divide chain steps into k groups to track progress
        t_start = time()
        for j in range(k):
            [self.take_step() for _ in range(m // k)]
            self.ProgressPrinter.percent_progress(t_start, j, k)

        # cleanup
        if m % k != 0:
            [self.take_step() for _ in range(m % k)]
        self.ProgressPrinter.percent_final(t_start, m)

    def run_for(self, minutes=0, hours=0, days=0):
        """
        Advances the chain for a chosen amount of computation time

        :param int minutes: number of minutes for which to run the chain.
        :param int hours: number of hours for which to run the chain.
        :param int days: number of days for which to run the chain.
        """
        update_interval = 20  # small initial guess for the update interval
        start_length = copy(self.chain_length)

        # first find the runtime in seconds:
        run_time = ((days * 24.0 + hours) * 60.0 + minutes) * 60.0
        start_time = time()
        current_time = start_time
        end_time = start_time + run_time

        while current_time < end_time:
            for i in range(update_interval):
                self.take_step()
            # set the interval such that updates are roughly once per second
            steps_taken = self.chain_length - start_length
            current_time = time()
            update_interval = int(steps_taken / (current_time - start_time))
            self.ProgressPrinter.countdown_progress(end_time, steps_taken)
        self.ProgressPrinter.countdown_final(run_time, steps_taken)

    @abstractmethod
    def get_parameter(self, index: int, burn: int = 0, thin: int = 1) -> ndarray:
        pass

    @abstractmethod
    def get_probabilities(self, burn: int = 0, thin: int = 1) -> ndarray:
        pass

    @abstractmethod
    def get_sample(self, burn: int = 0, thin: int = 1) -> ndarray:
        pass