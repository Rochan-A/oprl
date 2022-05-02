#############################
#
# Source: https://www.johndcook.com/blog/standard_deviation/,
# https://stackoverflow.com/a/45949178
#
#############################


from __future__ import division
import collections
import math

class RunningStats:
    def __init__(self, WIN_SIZE=20):
        self.n = 0
        self.mean = 0
        self.run_var = 0
        self.WIN_SIZE = WIN_SIZE

        self.windows = collections.deque(maxlen=WIN_SIZE)

    def clear(self):
        self.n = 0
        self.windows.clear()

    def push(self, x):

        self.windows.append(x)

        if self.n <= self.WIN_SIZE:
            # Calculating first variance
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.run_var += delta * (x - self.mean)
        else:
            # Adjusting variance
            x_removed = self.windows.popleft()
            old_m = self.mean
            self.mean += (x - x_removed) / self.WIN_SIZE
            self.run_var += (x + x_removed - old_m - self.mean) * (x - x_removed)

    def get_mean(self):
        return self.mean if self.n else 0.0

    def get_var(self):
        return self.run_var / (self.WIN_SIZE - 1) if self.n > 1 else 0.0

    def get_std(self):
        return math.sqrt(self.get_var())

    def get_all(self):
        return list(self.windows)

    def __str__(self):
        return "Current window values: {}".format(list(self.windows))
