"""
Grid Approximation
"""

import numpy as np


class GridApproximator:

    def __init__(self, low, high):
        """
        Args:
            low: Low bound
            high: High bound
        """

        self.low_bound = low
        self.high_bound = high

        # The grid
        self._grid = None
    
    @property
    def grid():
        if self._grid is not None:
            return self._grid
        else:
            raise TypeError("The grid is not created. Run create_uniform_grid()")

    def create_uniform_grid(bins=(10, 10)):
        """
        Define a uniformly-spaced grid that can be used to discretise a space

        Args:
            bins (tuple)
        """

        assert len(self.low_bound) == len(self.high_bound) == len(bins) 

        grid = [np.linspace(low_bound[d], high_bound[d], num=bins[d] + 1) for d in range(len(bins))]

        print("Uniform grid created: [low, high] / bins => Splits")
        for l, h, b, splits in zip(self.low_bound, self.high_bound, bins, grid):
            print("\t [%s, %s] / %s => %s" % (l, h, b, splits))
        
        self.grid = grid

    def discretise(sample):
        """
        Transform a x-y position to discretised ID
        e.g.
        [-1.0, -5.0] => [0, 0]

        Args:
            sample (array-like):
                A single sample from the original continuous spaceA
        
        Return:
            A discretised sample (array-like)
        """
        return list((np.digitise(s, g) for s, g in zip(sample, grid)))
