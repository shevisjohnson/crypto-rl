import unittest
import numpy as np
from scipy.special import softmax

from gym_trading.utils.portfolio import Portfolio
from configurations import FIAT, CRYPTOS, EXCHANGES, INITIAL_ALLOCATION

class PortfolioTestCases(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio(
            fiat            = FIAT,
            cryptos         = CRYPTOS,
            exchanges       = EXCHANGES,
            transaction_fee = True
        )
        vals1 = np.random.choice(100,len(self.portfolio.currencies))
        vals2 = np.random.choice(100,len(self.portfolio.currencies))
        inventory = dict(zip(self.portfolio.currencies, vals1))
        bid_prices = dict(zip(self.portfolio.currencies, vals2))
        self.portfolio.initialize(inventory, bid_prices)


    def test_step_updates_total_value(self):
        print(self.portfolio)
        vals2 = np.random.choice(100,len(self.portfolio.currencies))
        bid_prices = dict(zip(self.portfolio.currencies, vals2))
        self.portfolio.step(bid_prices)
        print(self.portfolio)

if __name__ == '__main__':
    unittest.main()