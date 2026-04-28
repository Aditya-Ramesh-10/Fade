import numpy as np

class LMS:
    """Least Mean Squares/Delta rule with fixed learning rate.
    Equivalent to SGD on MSE loss with no regularization."""
    def __init__(self, num_features, step_size=0.01, seed=0):

        self.random_state = np.random.RandomState(seed)
        self.weights = self.random_state.randn(num_features) * 0.1
        self.step_size = step_size

    def update_weights(self, x, y):
        y_pred = self.predict(x)
        error = y - y_pred
        self.weights += self.step_size * error * x
        return error ** 2

    def predict(self, x):
        return np.dot(self.weights, x)


class LMS_FixedWD:
    """LMS with fixed scalar weight decay."""
    def __init__(self, num_features, step_size=0.01,
                 weight_decay=0.001, seed=0):

        self.random_state = np.random.RandomState(seed)
        self.weights = self.random_state.randn(num_features) * 0.1
        self.step_size = step_size
        self.weight_decay = weight_decay
        
    def update_weights(self, x, y):
        y_pred = self.predict(x)
        error = y - y_pred
        self.weights = (1 - self.weight_decay) * self.weights + self.step_size * error * x
        return error ** 2

    def predict(self, x):
        return np.dot(self.weights, x)


class IDBD:
    """IDBD: Incremental Delta-Bar-Delta.
    Per-parameter adaptive learning rates, no weight decay."""
    def __init__(self, num_features, meta_step_size=0.0001,
                 initial_beta=np.log(0.01), seed=0):


        self.random_state = np.random.RandomState(seed)
        self.weights = self.random_state.randn(num_features) * 0.1

        self.beta = initial_beta * np.ones(num_features)

        self.memory_vector = np.zeros(num_features)
        self.meta_step_size = meta_step_size

    def update_weights(self, x, y):
        y_pred = self.predict(x)
        error = y - y_pred

        self.beta += self.meta_step_size * error * x * self.memory_vector

        step_size = np.exp(self.beta)

        self.weights += error * step_size * x

        self.memory_vector = (error * step_size * x) + np.clip(1 - (step_size * (x ** 2)), a_min=0, a_max=None) * self.memory_vector

        return error ** 2

    def predict(self, x):
        return np.dot(self.weights, x)

class FADE:
    """FADE: Forgetting through Adaptive DEcay.
    Fixed learning rate, per-parameter adaptive weight decay."""

    def __init__(self, num_features, lr=0.05, meta_step_size=0.001,
                 initial_gamma=-7.0, gamma_min=-np.inf,
                 gamma_max=np.inf, seed=0):
        self.random_state = np.random.RandomState(seed)
        self.weights = self.random_state.randn(num_features) * 0.1
        self.gamma = initial_gamma * np.ones(num_features)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.g = np.zeros(num_features)
        self.lr = lr
        self.meta_step_size = meta_step_size

    def update_weights(self, x, y):
        y_pred = self.predict(x)
        error = y - y_pred

        self.gamma += self.meta_step_size * error * x * self.g
        self.gamma = np.clip(self.gamma, self.gamma_min, self.gamma_max)
        decay = np.exp(self.gamma)

        self.g = (self.g * np.clip(1 - decay - self.lr * x ** 2, a_min=0, a_max=None)
                  - decay * self.weights)

        self.weights = (1 - decay) * self.weights + self.lr * error * x

        return error ** 2

    def predict(self, x):
        return np.dot(self.weights, x)


class IDBD_FixedWD:
    """IDBD with fixed scalar weight decay."""

    def __init__(self, num_features, meta_step_size=0.0001,
                 initial_beta=np.log(0.01), weight_decay=0.001, seed=0):
        self.random_state = np.random.RandomState(seed)
        self.weights = self.random_state.randn(num_features) * 0.1
        self.beta = initial_beta * np.ones(num_features)
        self.memory_vector = np.zeros(num_features)
        self.meta_step_size = meta_step_size
        self.wd = weight_decay

    def update_weights(self, x, y):
        y_pred = self.predict(x)
        error = y - y_pred

        self.beta += self.meta_step_size * error * x * self.memory_vector
        step_size = np.exp(self.beta)

        self.weights = (1 - self.wd) * self.weights + step_size * error * x

        self.memory_vector = (step_size * error * x
            + np.clip(1 - self.wd - step_size * x ** 2, a_min=0, a_max=None)
            * self.memory_vector)

        return error ** 2

    def predict(self, x):
        return np.dot(self.weights, x)


class FADE_IDBD:
    """FADE + IDBD: adaptive weight decay + adaptive learning rate."""

    def __init__(self, num_features, meta_step_size_alpha=0.001,
                 meta_step_size_lambda=0.001, initial_beta=np.log(0.01),
                 initial_gamma=-7.0, gamma_min=-np.inf, gamma_max=np.inf, seed=0):
        self.random_state = np.random.RandomState(seed)
        self.weights = self.random_state.randn(num_features) * 0.1
        self.beta = initial_beta * np.ones(num_features)
        self.gamma = initial_gamma * np.ones(num_features)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.h = np.zeros(num_features)
        self.g = np.zeros(num_features)
        self.meta_step_size_alpha = meta_step_size_alpha
        self.meta_step_size_lambda = meta_step_size_lambda

    def update_weights(self, x, y):
        y_pred = self.predict(x)
        error = y - y_pred

        self.beta += self.meta_step_size_alpha * error * x * self.h
        alpha = np.exp(self.beta)

        self.gamma += self.meta_step_size_lambda * error * x * self.g
        self.gamma = np.clip(self.gamma, self.gamma_min, self.gamma_max)
        decay = np.exp(self.gamma)

        clip = np.clip(1 - decay - alpha * x ** 2, a_min=0, a_max=None)
        self.h = self.h * clip + alpha * error * x
        self.g = self.g * clip - decay * self.weights

        self.weights = (1 - decay) * self.weights + alpha * error * x

        return error ** 2

    def predict(self, x):
        return np.dot(self.weights, x)


class CoupledWDSSAdaptation:
    """Adaptive weight decay + adaptive learning rate derived from L2 regularization."""
    
    def __init__(self, num_features, meta_step_size_alpha=0.001,
                 meta_step_size_lambda=0.001, initial_beta=np.log(0.01),
                 initial_gamma=-7.0, gamma_min=-np.inf, gamma_max=np.inf, seed=0):
        self.random_state = np.random.RandomState(seed)
        self.weights = self.random_state.randn(num_features) * 0.1
        self.beta = initial_beta * np.ones(num_features)
        self.gamma = initial_gamma * np.ones(num_features)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.h = np.zeros(num_features)
        self.g = np.zeros(num_features)
        self.meta_step_size_alpha = meta_step_size_alpha
        self.meta_step_size_lambda = meta_step_size_lambda

    def update_weights(self, x, y):
        y_pred = self.predict(x)
        error = y - y_pred

        self.beta += self.meta_step_size_alpha * error * x * self.h
        alpha = np.exp(self.beta)

        self.gamma += self.meta_step_size_lambda * error * x * self.g
        self.gamma = np.clip(self.gamma, self.gamma_min, self.gamma_max)
        decay = np.exp(self.gamma)

        clip = np.clip(1 - alpha *decay - alpha * x ** 2, a_min=0, a_max=None)
        self.h = self.h * clip + alpha * error * x - alpha * decay * self.weights
        self.g = self.g * clip - alpha * decay * self.weights

        self.weights = (1 - alpha *decay) * self.weights + alpha * error * x

        return error ** 2

    def predict(self, x):
        return np.dot(self.weights, x)