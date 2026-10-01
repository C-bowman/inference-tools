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
        self.midpoint = 0.5 * (lower + upper)
        self.scale = 0.25 * self.width
        self.inv_scale = 4.0 / self.width
        self.ln_4 = log(4.0)

    def forward(self, value: ndarray) -> ndarray:
        if ((value <= self.lower) | (value >= self.upper)).any():
            raise ValueError("IntervalTransform values must lie strictly inside the bounds")
        logit = log(value - self.lower) - log(self.upper - value)
        return self.midpoint + self.scale * logit

    def inverse(self, value: ndarray) -> ndarray:
        normalized = (value - self.midpoint) * self.inv_scale
        return self.lower + self.width * expit(normalized)

    def log_prob_inputs(self, value: ndarray) -> tuple[ndarray, float]:
        normalized = (value - self.midpoint) * self.inv_scale
        sigmoid = expit(normalized)
        constrained = self.lower + self.width * sigmoid
        elementwise = (
            self.ln_4
            - logaddexp(0.0, -normalized)
            - logaddexp(0.0, normalized)
        )
        return constrained, float(elementwise.sum())

    def log_jacobian(self, value: ndarray) -> float:
        normalized = (value - self.midpoint) * self.inv_scale
        elementwise = (
            self.ln_4
            - logaddexp(0.0, -normalized)
            - logaddexp(0.0, normalized)
        )
        return float(elementwise.sum())

    def gradient_inputs(self, value: ndarray) -> tuple[ndarray, ndarray, ndarray]:
        normalized = (value - self.midpoint) * self.inv_scale
        sigmoid = expit(normalized)
        constrained = self.lower + self.width * sigmoid
        jacobian = 4.0 * sigmoid * (1.0 - sigmoid)
        log_jacobian_gradient = self.inv_scale * (1.0 - 2.0 * sigmoid)
        return constrained, jacobian, log_jacobian_gradient
