import unittest
import numpy as np
from scipy.special import softmax
from copy import deepcopy

from gym_trading.utils.portfolio import Portfolio
from gym_trading.utils.order import MarketOrder
from configurations import FIAT, CRYPTOS, EXCHANGES, INITIAL_ALLOCATION

class PortfolioTestCases(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio(
            fiat            = FIAT,
            cryptos         = CRYPTOS,
            exchanges       = EXCHANGES,
            transaction_fee = True
        )

    def random_init(self):
        vals1 = np.random.choice(10000,len(self.portfolio.currencies)) / 10.0
        vals2 = np.random.choice(10000,len(self.portfolio.currencies)) / 10.0
        inventory = dict(zip(self.portfolio.currencies, vals1))
        bid_prices = dict(zip(self.portfolio.currencies, vals2))
        self.portfolio.initialize(bid_prices, inventory)

    def test_step_updates_values(self):
        self.random_init()
        total_value_before_step = self.portfolio.total_value
        pnl_before_step = self.portfolio.pnl
        for i in range(5):
            vals2 = np.random.choice(10000,len(self.portfolio.currencies)) / 10.0
            bid_prices = dict(zip(self.portfolio.currencies, vals2))
            self.portfolio.step(bid_prices)
        total_value_after_step = self.portfolio.total_value
        pnl_after_step = self.portfolio.pnl
        self.assertNotEqual(total_value_before_step, total_value_after_step)
        self.assertNotEqual(pnl_before_step, pnl_after_step)


    def test_add_order_modifies_portfolio_attributes(self):
        bp = deepcopy(self.portfolio.bid_prices)
        bp['BTC'] = 11000.0
        inv = deepcopy(self.portfolio.inventory)
        inv['USD'] = 11000.0
        self.portfolio.initialize(bp, inv)

        total_value_before_order = self.portfolio.total_value
        pnl_before_order = self.portfolio.pnl
        realized_value_before_order = self.portfolio.realized_value
        realized_pnl_before_order = self.portfolio.realized_pnl
        
        order = MarketOrder(ccy='BTC-USD', side='long', price=11000.0, size=1.0)

        self.assertTrue(self.portfolio.add_order(order))

        total_value_after_order = self.portfolio.total_value
        pnl_after_order = self.portfolio.pnl
        realized_value_after_order = self.portfolio.realized_value
        realized_pnl_after_order = self.portfolio.realized_pnl

        self.assertNotEqual(total_value_before_order, total_value_after_order)
        self.assertNotEqual(pnl_before_order, pnl_after_order)
        self.assertNotEqual(realized_value_before_order, realized_value_after_order)
        self.assertNotEqual(realized_pnl_before_order, realized_pnl_after_order)

if __name__ == '__main__':
    unittest.main()