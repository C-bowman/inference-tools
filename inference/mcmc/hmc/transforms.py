from numpy import ndarray, log, logaddexp
from scipy.special import expit


class IdentityTransform:
    def forward(self, value: ndarray) -> ndarray:
        return value.copy()

    def inverse(self, value: ndarray) -> ndarray:
        return value.copy()

    def log_prob_inputs(self, value: ndarray) -> tuple[ndarray, float]:
        return value, 0.0

    def log_jacobian(self, value: ndarray) -> float:
        return 0.0

    def gradient_inputs(self, value: ndarray) -> tuple[ndarray, float, float]:
        return value, 1.0, 0.0


class IntervalTransform:
    def __init__(self, lower: ndarray, upper: ndarray):
        self.lower = lower
        self.upper = upper
        self.width = upper - lower
        self.log_width = log(self.width)

    def forward(self, value: ndarray) -> ndarray:
        if ((value <= self.lower) | (value >= self.upper)).any():
            raise ValueError("IntervalTransform values must lie strictly inside the bounds")
        return log(value - self.lower) - log(self.upper - value)

    def inverse(self, value: ndarray) -> ndarray:
        return self.lower + self.width * expit(value)

    def log_prob_inputs(self, value: ndarray) -> tuple[ndarray, float]:
        sigmoid = expit(value)
        constrained = self.lower + self.width * sigmoid
        return constrained, self.log_jacobian(value)

    def log_jacobian(self, value: ndarray) -> float:
        elementwise = (
            self.log_width - logaddexp(0.0, -value) - logaddexp(0.0, value)
        )
        return float(elementwise.sum())

    def gradient_inputs(self, value: ndarray) -> tuple[ndarray, ndarray, ndarray]:
        sigmoid = expit(value)
        constrained = self.lower + self.width * sigmoid
        jacobian = self.width * sigmoid * (1.0 - sigmoid)
        log_jacobian_gradient = 1.0 - 2.0 * sigmoid
        return constrained, jacobian, log_jacobian_gradient
