import numpy as np
import torch

class PeriodicLinearParameterFlip:

    def __init__(self, period=20, num_total_components=20, num_flipping_components=5,
                 noise_level=0.0, seed=0):
        self.period = period
        self.num_total_components = num_total_components
        self.num_flipping_components = num_flipping_components
        self.noise_level = noise_level
        self.random_state = np.random.RandomState(seed)

        self.weights = np.zeros(num_total_components)

        weight_sample = self.random_state.choice([-1, 1], num_flipping_components)
        self.weights[:num_flipping_components] = weight_sample

        self.time_step = 0

    def sample_training_example(self):
        x = self.random_state.randn(self.num_total_components)
        y = np.dot(self.weights, x)

        y += self.random_state.randn() * self.noise_level

        self.time_step += 1
        if self.time_step % self.period == 0:
            self._flip_random_weight_components()

        return x, y

    def _flip_random_weight_components(self):
        flip_component = self.random_state.choice(self.num_flipping_components)
        self.weights[flip_component] *= -1
