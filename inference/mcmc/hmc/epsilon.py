from collections import defaultdict
from copy import copy
from numpy import sqrt, log, exp, array

from scipy.special import ndtr as normal_cdf


class RunningStats:
    def __init__(self):
        self.mean: float = 0.0
        self.S: float = 0.0
        self.variance: float = 0.0
        self.count: int = 0
        self.update = self.first_sample

    def first_sample(self, x):
        self.count = 1
        self.mean = x
        self.update = self.add_sample

    def add_sample(self, x):
        self.count += 1
        mu = self.mean + (x - self.mean) / self.count
        self.S += (x - self.mean) * (x - mu)
        self.mean = mu
        self.variance = self.S / (self.count - 1)


class EpsilonSelector:
    def __init__(self, initial_epsilon: float):
        # The spacing of the discretisation of the log-epsilon value
        self.ln_eps_spacing = log(1.02)
        # the factor by which epsilon is increased or decreased when searching
        self.search_factor = 2.0
        # the number of indices to jump in order to achieve the desired search factor
        self.index_jump = int(round(log(self.search_factor) / self.ln_eps_spacing))
        # round the given epsilon initial guess to the nearest discretised value
        self.current_index = round(log(initial_epsilon) / self.ln_eps_spacing)
        self.epsilon = exp(self.ln_eps_spacing * self.current_index)

        # storage to record history of changes in epsilon
        self.epsilon_values = [copy(self.epsilon)]
        self.epsilon_updates = [0]

        # tracking of mean / variance of acceptance rate at each epsilon
        self.stats = defaultdict(RunningStats)
        self.counter = 0
        self.total_proposals = 0

        # target acceptance rate
        self.accept_rate = 0.65
        # interval of steps at which proposal widths are adjusted
        self.update_interval = 25
        self.min_var = 1e-3

    def add_probability(self, p: float):
        self.stats[self.current_index].update(p)
        self.counter += 1

        if self.counter >= self.update_interval:
            self.total_proposals += self.counter
            self.counter = 0
            self.update_epsilon()

            if self.epsilon != self.epsilon_values[-1]:
                self.epsilon_values.append(self.epsilon)
                self.epsilon_updates.append(self.total_proposals)

    def update_epsilon(self):
        bin_inds = array([i for i in self.stats.keys()])
        bin_probs = array(
            [self.below_target_probability(s) for s in self.stats.values()]
        )

        sorter = bin_inds.argsort()
        bin_inds = bin_inds[sorter]
        bin_probs = bin_probs[sorter]
        bin_probs = bin_probs.clip(1e-4, 1 - 1e-4)

        interval_probs = [bin_probs.prod()]
        for p in bin_probs:
            new = interval_probs[-1] * (1 - p) / p
            interval_probs.append(new)

        interval_probs = array(interval_probs)
        max_ind = interval_probs.argmax()

        if max_ind == 0:
            self.current_index = bin_inds[0] - self.index_jump
        elif max_ind == interval_probs.size - 1:
            self.current_index = bin_inds[-1] + self.index_jump
        else:
            # get the indices of the two bin edges
            lwr = bin_inds[max_ind - 1]
            upr = bin_inds[max_ind]

            # if the bins are adjacent, we calculate the one with the highest
            # likelihood of being equal to the target rate
            if upr - lwr == 1:
                lwr_prob = self.target_likelihood(self.stats[lwr])
                upr_prob = self.target_likelihood(self.stats[upr])
                self.current_index = upr if upr_prob >= lwr_prob else lwr

            # otherwise, aim for a new bin in the middle of the two
            else:
                self.current_index = int(round((lwr + upr) * 0.5))

        # calculate new epsilon based on new index
        self.epsilon = exp(self.ln_eps_spacing * self.current_index)

    def below_target_probability(self, stats: RunningStats) -> float:
        """
        Calculates the probability that the true acceptance rate for the epsilon value
        corresponding to the given `RunningStats` instance is below the target
        acceptance rate by assuming a gaussian uncertainty on the mean of the observed
        acceptance rate
        """
        inv_sigma = sqrt(stats.count / max(stats.variance, self.min_var))
        z = (self.accept_rate - stats.mean) * inv_sigma
        return normal_cdf(z)

    def target_likelihood(self, stats: RunningStats) -> float:
        inv_sigma = sqrt(stats.count / max(stats.variance, self.min_var))
        z = (self.accept_rate - stats.mean) * inv_sigma
        return exp(-0.5 * z**2) * inv_sigma

    def get_items(self):
        items = self.__dict__.copy()
        items.pop("stats")

        items["stats_mean"] = [rs.mean for rs in self.stats.values()]
        items["stats_variance"] = [rs.variance for rs in self.stats.values()]
        items["stats_s"] = [rs.S for rs in self.stats.values()]
        items["stats_counts"] = [rs.count for rs in self.stats.values()]
        items["stats_index"] = [i for i in self.stats.keys()]
        return items

    def load_items(self, dictionary: dict):
        self.epsilon = float(dictionary["epsilon"])
        self.epsilon_values = list(dictionary["epsilon_values"])
        self.epsilon_updates = list(dictionary["epsilon_updates"])
        self.accept_rate = float(dictionary["accept_rate"])
        self.update_interval = int(dictionary["update_interval"])
        self.counter = int(dictionary["counter"])
        self.total_proposals = int(dictionary["total_proposals"])
        self.ln_eps_spacing = float(dictionary["ln_eps_spacing"])
        self.search_factor = float(dictionary["search_factor"])
        self.current_index = int(dictionary["current_index"])

        # rebuild the dictionary of RunningStats objects
        self.stats = {}
        for k, index in enumerate(dictionary["stats_index"]):
            rs = RunningStats()
            rs.mean = dictionary["stats_mean"][k]
            rs.variance = dictionary["stats_variance"][k]
            rs.s = dictionary["stats_s"][k]
            rs.count = int(dictionary["stats_counts"][k])
            if rs.count > 0:
                rs.update = rs.add_sample
            self.stats[index] = rs
