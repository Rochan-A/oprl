from os import stat
import numpy as np
import pdb
from typing import List, Dict
import random
import math
from scipy.special import softmax
from scipy import stats

def categorical_draw(probs):
    z = random.random()
    cum_prob = 0.0
    for i in range(len(probs)):
        prob = probs[i]
        cum_prob += prob
        if cum_prob > z:
            return i
    return len(probs) - 1


class ExpWeights(object):
    def __init__(
        self,
        N: int,
        lr: float = 0.2,
        window: int = 5,
        decay: float = 0.9,
        init: float = 1.0,
        use_std: bool = False,
    ) -> None:
        """Initialize bandit.
        Args:
                N (int): # of arms
                lr (float, optional): Learning rate. Defaults to 0.2.
                window (int, optional): Window to normalize over. Defaults to 5.
                decay (float, optional): Decay rate for probability. Defaults to 0.9.
                init (float, optional): Weight initialization. Defaults to 0.0.
                use_std (bool, optional): Use std to normalize feedback. Defaults to True.
        """

        self.arms = [i for i in range(N)]
        self.w = {i: init for i in range(len(self.arms))}
        self.arm = 0
        self.value = self.arms[self.arm]
        self.error_buffer = []
        self.window = window
        self.lr = lr
        self.decay = decay
        self.use_std = use_std

        self.choices = [self.arm]
        self.data = []

    def sample(self) -> float:
        """Sample from distribution.
        Returns:
                float: The value of the sampled arm.
        """
        p = [x for x in self.w.values()]
        if np.isnan(softmax(p)).any():
            p = softmax(np.nan_to_num(p, nan=0.01))
        else:
            p = softmax(p)
        self.arm = np.random.choice(range(0, len(p)), p=p)

        self.value = self.arms[self.arm]
        self.choices.append(self.arm)

        return self.value

    def get_probs(self) -> List:
        """Get arm probabilities.
        Returns:
                List: probabilities for each arm.
        """
        p = [x for x in self.w.values()]
        if np.isnan(softmax(p)).any():
            p = softmax(np.nan_to_num(p, nan=0.01))
        else:
            p = softmax(p)
        return p

    def get_values(self) -> List:
        """Return the underlying values for each arm
        Returns:
                List: np.exp(x) for x in self.w.values()
        """
        return [np.exp(x) for x in self.w.values()]

    def update(self, feedback: float, norm: float = 1.0) -> None:
        """Update distribution over arms.
        Args:
                feedback (float): Feedback signal.
                norm (float, optional): Normalization factor. Defaults to 1.0.
        """

        # Since this is non-stationary, subtract mean of previous self.window errors.
        self.error_buffer.append(feedback)
        self.error_buffer = self.error_buffer[-self.window :]

        # normalize
        feedback -= np.mean(self.error_buffer)
        if self.use_std and len(self.error_buffer) > 1:
            norm = np.clip(np.std(self.error_buffer), 0.001, None)
        feedback /= norm

        # update arm weights
        self.w[self.arm] *= self.decay
        self.w[self.arm] += self.lr * (feedback / max(np.exp(self.w[self.arm]), 0.0001))

        self.data.append(feedback)


class ExpWeightsProxG(object):
    def __init__(
        self,
        N: int,
        lr: float = 0.2,
        window: int = 5,
        decay: float = 0.9,
        init: float = 1.0,
        use_std: bool = False,
        prox: str = 'linear'
    ) -> None:
        """Initialize bandit.
        Args:
                N (int): # of arms
                lr (float, optional): Learning rate. Defaults to 0.2.
                window (int, optional): Window to normalize over. Defaults to 5.
                decay (float, optional): Decay rate for probability. Defaults to 0.9.
                init (float, optional): Weight initialization. Defaults to 0.0.
                use_std (bool, optional): Use std to normalize feedback. Defaults to True.
        """

        self.arms = [i for i in range(N)]
        self.w = {i: init for i in range(len(self.arms))}
        self.arm = 0
        self.value = self.arms[self.arm]
        self.error_buffer = []
        self.window = window
        self.lr = lr
        self.decay = decay
        self.use_std = use_std

        self.choices = [self.arm]
        self.data = []

    def sample(self) -> float:
        """Sample from distribution.
        Returns:
                float: The value of the sampled arm.
        """
        p = [x for x in self.w.values()]
        if np.isnan(softmax(p)).any():
            p = softmax(np.nan_to_num(p, nan=0.01))
        else:
            p = softmax(p)
        self.arm = np.random.choice(range(0, len(p)), p=p)

        self.value = self.arms[self.arm]
        self.choices.append(self.arm)

        return self.value

    def get_probs(self) -> List:
        """Get arm probabilities.
        Returns:
                List: probabilities for each arm.
        """
        p = [x for x in self.w.values()]
        if np.isnan(softmax(p)).any():
            p = softmax(np.nan_to_num(p, nan=0.01))
        else:
            p = softmax(p)
        return p

    def get_values(self) -> List:
        """Return the underlying values for each arm
        Returns:
                List: np.exp(x) for x in self.w.values()
        """
        return [np.exp(x) for x in self.w.values()]

    def update(self, feedback: float, norm: float = 1.0) -> None:
        """Update distribution over arms.
        Args:
                feedback (float): Feedback signal.
                norm (float, optional): Normalization factor. Defaults to 1.0.
        """

        # Since this is non-stationary, subtract mean of previous self.window errors.
        self.error_buffer.append(feedback)
        self.error_buffer = self.error_buffer[-self.window :]

        # normalize
        feedback -= np.mean(self.error_buffer)
        if self.use_std and len(self.error_buffer) > 1:
            norm = np.clip(np.std(self.error_buffer), 0.001, None)
        feedback /= norm

        # update arm weights
        self.w[self.arm] *= self.decay
        self.w[self.arm] += self.lr * (feedback / max(np.exp(self.w[self.arm]), 0.0001))

        self.data.append(feedback)


class Exp3:
    def __init__(self, N, gamma):
        self.gamma = gamma
        self.initialize(N)
        self.arm = 0

    def initialize(self, n_arms):
        self.weights = [1.0 for i in range(n_arms)]

    def sample(self):
        n_arms = len(self.weights)
        total_weight = sum(self.weights)
        probs = [0.0 for i in range(n_arms)]
        for arm in range(n_arms):
            probs[arm] = (1 - self.gamma) * (self.weights[arm] / total_weight)
            probs[arm] = probs[arm] + (self.gamma) * (1.0 / float(n_arms))
        self.arm = categorical_draw(probs)
        return self.arm + 1

    def update(self, reward):
        chosen_arm = self.arm
        n_arms = len(self.weights)
        total_weight = sum(self.weights)
        probs = [0.0 for i in range(n_arms)]
        for arm in range(n_arms):
            probs[arm] = (1 - self.gamma) * (self.weights[arm] / total_weight)
            probs[arm] = probs[arm] + (self.gamma) * (1.0 / float(n_arms))

        x = reward / probs[chosen_arm]

        growth_factor = math.exp((self.gamma / n_arms) * x)
        self.weights[chosen_arm] = self.weights[chosen_arm] * growth_factor


    def get_probs(self):
        n_arms = len(self.weights)
        total_weight = sum(self.weights)
        probs = [0.0 for i in range(n_arms)]
        for arm in range(n_arms):
            probs[arm] = (1 - self.gamma) * (self.weights[arm] / total_weight)
            probs[arm] = probs[arm] + (self.gamma) * (1.0 / float(n_arms))
        return probs


    def get_values(self):
        return self.weights


class UCB(object):
    def __init__(self, N, lr, c=2) -> None:
        self.q_est = np.zeros(N)
        self.num_a_est = np.ones(N)
        self.lr = lr
        self.c = c
        self.t = 1

        self.active_estimator = np.random.randint(N)

    def update(self, reward):
        self.q_est[self.active_estimator] += self.lr * (reward - self.q_est[self.active_estimator])
        self.num_a_est[self.active_estimator] += 1
        self.t += 1

    def get_probs(self):
        return self.q_est + self.c * np.sqrt(np.log(self.t) / (self.num_a_est))

    def get_values(self):
        return self.q_est

    def sample(self):
        return np.argmax(self.get_probs()) + 1


class ThompsonSamplingNonStationary(object):
    def __init__(
        self,
        N,
        window: int = 5,
        gamma: float = 0.9,
        alpha: float = 1.0,
        beta: bool = 3.0,
    ) -> None:
        """Initialize bandit.
        Args:
            N (int): # of arms
            window (int, optional): Window to normalize over. Defaults to 5.
            gamma (float, optional): Decay rate for probability. Defaults to 0.9.
            alpha (float, optional): Weight initialization. Defaults to 0.0.
            beta (float, optional): Use std to normalize feedback. Defaults to True.
        """

        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.error_buffer = []
        self.window = window
        self.S = np.zeros((self.N))
        self.F = np.zeros((self.N))
        
        self.theta = np.zeros((self.N))
        for i in range(self.N):
            self.theta[i] = stats.beta.rvs(self.S[i] + self.alpha, self.F[i] + self.beta, size=1)

    def sample(self) -> float:
        """Sample from distribution.
        Returns:
                float: The value of the sampled arm.
        """
        self.arm = np.argmax(self.theta)
        return self.arm+1  # arms indexed from 1

    def get_probs(self) -> List:
        """Get arm probabilities.
        Returns:
                List: probabilities for each arm.
        """
        return self.theta

    def get_values(self) -> List:
        """Return the underlying values for each arm
        Returns:
                List: np.exp(x) for x in self.w.values()
        """
        return self.theta

    def update(self, feedback: float, norm: float = 1.0) -> None:
        """Update distribution over arms.
        Args:
                feedback (float): Feedback signal. Has to be in range [0, 1]!
                norm (float, optional): Normalization factor. Defaults to 1.0.
        """
        # Since this is non-stationary, subtract mean of previous self.window errors.
        self.error_buffer.append(feedback)
        self.error_buffer = self.error_buffer[-self.window :]

        # normalize
        feedback -= np.mean(self.error_buffer)
        if len(self.error_buffer) > 1:
            if np.std(self.error_buffer) > 0:
                norm = np.std(self.error_buffer)
                feedback /= norm
            else:
                feedback += np.mean(self.error_buffer)

        r = stats.bernoulli.rvs(np.clip(feedback, 0, 1), 1)
        self.S[self.arm] = self.gamma*self.S[self.arm] + r
        self.F[self.arm] = self.gamma*self.F[self.arm] + (1 - r)
        
        idx = np.delete(np.arange(self.N), self.arm)
        self.S[idx] = self.gamma*self.S[idx]
        self.F[idx] = self.gamma*self.F[idx]
        for i in range(self.N):
            self.theta[i] = stats.beta.rvs(np.clip(self.S[i] + self.alpha, 0.01, None), np.clip(self.F[i] + self.beta, 0.01, None), size=1)


if __name__ == "__main__":
    # TDC = ExpWeights(arms=[0, 1, 2, 3, 4], lr=0.1, init=1.0, use_std=True)

    # for i in range(50):
    #     print(TDC.get_probs(), TDC.sample())
    #     TDC.update_dists(-np.random.rand(), 0.1)
    #     print()

    TDC = ThompsonSamplingNonStationary(N=5, window=10, gamma=0.1, alpha=1, beta=1)

    for i in range(50):
        print(TDC.get_probs(), TDC.sample())
        TDC.update(np.random.rand(), 1)
        print()
