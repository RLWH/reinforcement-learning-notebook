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

        self.low = low
        self.high = high