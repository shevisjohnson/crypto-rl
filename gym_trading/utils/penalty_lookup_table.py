import pandas as pd
import numpy as np

from configurations import MARKET_ORDER_FEE, ENCOURAGEMENT, CURRENCIES, GRIDMAX_LEVEL

class PenaltyLookupTable(object):
    """
    This class generates an efficient way to lookup the penalty incurred
    when moving from one portfolio allocation to another.

    :param n_currencies: (int) number of currencies in portfolio
    :param gridmax_level: (int) gridmax level to generate descrete portfolio allocations
    """
    def __init__(self,
                 allocations: np.ndarray):
        self.allocations = allocations
        self._warmed_up = False

    def warm_up(self) -> None:
        # np.roll(self.allocations, i, axis=0)
        allocations_squared = np.dstack(
            [np.copy(self.allocations)
             for i in range(self.allocations.shape[0])]
        ).T
        allocations_squared = np.array([allocations_squared[i].T for i in range(self.allocations.shape[0])], dtype=np.float32)
        self.table = np.ndarray((self.allocations.shape[0], self.allocations.shape[0]))
        for i in range(self.allocations.shape[0]):
            curr = allocations_squared[i]
            base = curr[i]
            for j in range(self.allocations.shape[0]):
                target = curr[j]
                allocation_diff = target - base
                difference_factor = np.float32(np.sum(np.absolute(allocation_diff), dtype=np.float32) / np.float32(2))
                df_compliment = np.float32(1) - difference_factor
                self.table[i][j] = (df_compliment + (difference_factor * (np.float32(1) - MARKET_ORDER_FEE)) + ENCOURAGEMENT) - np.float32(1)
        self._warmed_up = True

    def cool_down(self):
        if self._warmed_up:
            del self.table
            self._warmed_up = False

    def get_table_idx_for_allocation(self, allocation: np.ndarray) -> np.ndarray:
        if not self._warmed_up: self.warm_up()
        closest_grid = self.allocations[np.argmin(np.array([np.linalg.norm(allocation-x) for x in self.allocations]))]
        return np.flatnonzero((self.allocations == closest_grid).all(1))[0]

    def get_penalty_for_action(self, allocation: np.ndarray, target: np.ndarray) -> np.float32:
        allocation_idx = self.get_table_idx_for_allocation(allocation)
        target_idx = self.get_table_idx_for_allocation(target)
        return self.table[allocation_idx][target_idx]


