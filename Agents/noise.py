import numpy as np
import copy

class OUActionNoise():
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def __call__(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state
        
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

class NormalNoise():
    def __init__(self, normal_scalar):
        self.normal_scalar = normal_scalar
    
    def __call__(self):
        return np.random.randn(1) * self.normal_scalar
    



