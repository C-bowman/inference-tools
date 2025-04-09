from copy import copy
from numpy import sqrt, log


class EpsilonSelector:
    def __init__(self, epsilon: float):
        # storage
        self.epsilon = epsilon
        self.epsilon_values = [copy(epsilon)]  # sigma values after each assessment
        self.epsilon_checks = [0.0]  # chain locations at which sigma was assessed

        # tracking variables
        self.avg = 0
        self.var = 0
        self.num = 0

        # settings for epsilon adjustment algorithm
        self.accept_rate = 0.65
        self.chk_int = 15  # interval of steps at which proposal widths are adjusted
        self.growth_factor = 1.4  # growth factor for self.chk_int

    def add_probability(self, p: float):
        self.num += 1
        self.avg += p
        self.var += max(p * (1 - p), 0.03)

        if self.num >= self.chk_int:
            self.update_epsilon()

    def update_epsilon(self):
        """
        looks at the acceptance rate of proposed steps and adjusts the epsilon
        value to bring the acceptance rate toward its target value.
        """
        # normal approximation of poisson binomial distribution
        mu = self.avg / self.num
        std = sqrt(self.var) / self.num

        # now check if the desired success rate is within 2-sigma
        if ~(mu - 2 * std < self.accept_rate < mu + 2 * std):
            adj = (log(self.accept_rate) / log(mu)) ** 0.15
            adj = min(adj, 2.0)
            adj = max(adj, 0.5)
            self.adjust_epsilon(adj)
        else:  # increase the check interval
            self.chk_int = int((self.growth_factor * self.chk_int) * 0.1) * 10

    def adjust_epsilon(self, ratio: float):
        self.epsilon *= ratio
        self.epsilon_values.append(copy(self.epsilon))
        self.epsilon_checks.append(self.epsilon_checks[-1] + self.num)
        self.avg = 0
        self.var = 0
        self.num = 0

    def get_items(self):
        return self.__dict__

    def load_items(self, dictionary: dict):
        self.epsilon = float(dictionary["epsilon"])
        self.epsilon_values = list(dictionary["epsilon_values"])
        self.epsilon_checks = list(dictionary["epsilon_checks"])
        self.avg = float(dictionary["avg"])
        self.var = float(dictionary["var"])
        self.num = float(dictionary["num"])
        self.accept_rate = float(dictionary["accept_rate"])
        self.chk_int = int(dictionary["chk_int"])
        self.growth_factor = float(dictionary["growth_factor"])
