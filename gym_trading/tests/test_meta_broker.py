import unittest
import numpy as np
from scipy.special import softmax
from copy import deepcopy
from pprint import pprint

from gym_trading.utils.portfolio import Portfolio
from gym_trading.utils.order import MarketOrder
from gym_trading.utils.meta_broker import MetaBroker
from configurations import FIAT, CRYPTOS, EXCHANGES, INITIAL_ALLOCATION


class MetaBrokerTestCases(unittest.TestCase):
    def setUp(self):
        self.broker = MetaBroker(
            fiat            = FIAT,
            cryptos         = CRYPTOS,
            exchanges       = EXCHANGES,
            transaction_fee = True
        )

    def random_init(self):
        vals1 = np.random.choice(10000,len(self.broker.portfolio.currencies)) / 10.0
        vals2 = np.random.choice(10000,len(self.broker.portfolio.currencies)) / 10.0
        inventory = dict(zip(self.broker.portfolio.currencies, vals1))
        bid_prices = dict(zip(self.broker.portfolio.currencies, vals2))
        self.broker.initialize(bid_prices, inventory)

    def test_step_updates_portfolio_attributes(self):
        self.random_init()
        total_value_before_step = self.broker.portfolio.total_value
        pnl_before_step = self.broker.portfolio.pnl
        for i in range(5):
            vals2 = np.random.choice(10000,len(self.broker.portfolio.currencies)) / 10.0
            bid_prices = dict(zip(self.broker.portfolio.currencies, vals2))
            self.broker.portfolio.step(bid_prices)
        total_value_after_step = self.broker.portfolio.total_value
        pnl_after_step = self.broker.portfolio.pnl
        self.assertNotEqual(total_value_before_step, total_value_after_step)
        self.assertNotEqual(pnl_before_step, pnl_after_step)

    def test_add_order_updates_portfolio_attributes(self):
        bp = deepcopy(self.broker.portfolio.bid_prices)
        bp['BTC'] = 11000.0
        inv = deepcopy(self.broker.portfolio.inventory)
        inv['USD'] = 11000.0
        self.broker.initialize(bp, inv)

        total_value_before_order = self.broker.portfolio.total_value
        pnl_before_order = self.broker.portfolio.pnl
        realized_value_before_order = self.broker.portfolio.realized_value
        realized_pnl_before_order = self.broker.portfolio.realized_pnl
        
        order = MarketOrder(ccy='BTC-USD', side='long', price=11000.0, size=1.0)

        self.assertTrue(self.broker.portfolio.add_order(order))

        total_value_after_order = self.broker.portfolio.total_value
        pnl_after_order = self.broker.portfolio.pnl
        realized_value_after_order = self.broker.portfolio.realized_value
        realized_pnl_after_order = self.broker.portfolio.realized_pnl

        self.assertNotEqual(total_value_before_order, total_value_after_order)
        self.assertNotEqual(pnl_before_order, pnl_after_order)
        self.assertNotEqual(realized_value_before_order, realized_value_after_order)
        self.assertNotEqual(realized_pnl_before_order, realized_pnl_after_order)

    def test_get_statistics(self):
        bp = deepcopy(self.broker.portfolio.bid_prices)
        bp['BTC'] = 10000.0
        inv = deepcopy(self.broker.portfolio.inventory)
        inv['USD'] = 20000.0
        self.broker.initialize(bp, inv)
        
        order = MarketOrder(ccy='BTC-USD', side='long', price=10000.0, size=1.0)

        self.assertTrue(self.broker.portfolio.add_order(order))

        bp['BTC'] = 10100.0
        self.broker.step(bp)

        expected_stats = {
            'final_portfolio_value': '$20049.50',
            'initial_portfolio_value': '$20000.00',
            'market_orders': 1,
            'notional_pnl': '+0.248%',
            'realized_pnl': '-50.000%',
        }

        actual_stats = self.broker.get_statistics()
        pprint(actual_stats)
        self.assertDictEqual(expected_stats, actual_stats)

if __name__ == '__main__':
    unittest.main()