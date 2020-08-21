import unittest
import numpy as np
from scipy.special import softmax
from copy import deepcopy
from pprint import pprint

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
        currency_amounts = np.random.choice(10000,len(self.broker.portfolio.currencies)) / 10.0
        exchange_rates = np.random.choice(10000,len(self.broker.portfolio.exchanges)) / 10.0
        self.final_init(currency_amounts, exchange_rates)

    def zero_init(self):
        currency_amounts = np.zeros(len(self.broker.portfolio.currencies))
        exchange_rates = np.zeros(len(self.broker.portfolio.exchanges))
        self.final_init(currency_amounts, exchange_rates)

    def final_init(self, currency_amounts, exchange_rates):
        self.inventory = dict(zip(self.broker.portfolio.currencies, currency_amounts))
        self.bid_ask_prices = dict(zip(self.broker.portfolio.exchanges,
                                  [{'ask': v * 1.00001, 'bid': v * 0.99999} for v in exchange_rates]))
        self.broker.initialize(self.bid_ask_prices, self.inventory)

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
        self.zero_init()
        bp = deepcopy(self.bid_ask_prices)
        bp['BTC-USD'] = {'bid': 11000.0, 'ask': 11000.01}
        inv = deepcopy(self.inventory)
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

    def run_single_reallocation_test(self):
        exchange_rates = [
            11812.31,
            411.30,
            15.22256,
            0.105917,
            0.03478,
            0.00000898,
            0.03709380,
        ]
        bid_ask_prices = dict(zip(EXCHANGES,
                                  [{'ask': v * 1.00001, 'bid': v * 0.99999} for v in exchange_rates]))
        inventory = {}
        for c in self.broker.portfolio.currencies:
            if c == 'USD':
                inventory[c] = 10000.0
            else:
                inventory[c] = 10000.0 / bid_ask_prices[f'{c}-USD']['bid']

        self.broker.initialize(bid_ask_prices, inventory)

        initial_allocation = self.broker.allocation

        currency_range = range(len(self.broker.portfolio.currencies))
        
        nums = np.array([np.random.uniform() for _ in currency_range])
        total = np.sum(nums)

        target_allocation = {self.broker.portfolio.currencies[i]: nums[i] / total
                             for i in currency_range}

        #print()
        #print("BEFORE", self.broker.allocation)
        #print("TARGET", target_allocation)
        reached_target = self.broker.reallocate(target_allocation)
        #print("AFTER", self.broker.allocation)
        #print(self.broker.portfolio)
        return reached_target

    def test_reallocation(self):
        cts = {True: 0.0, False: 0.0}
        for _ in range(100):
            reached_target = self.run_single_reallocation_test()
            self.broker.reset()
            cts[reached_target] += 1.0

        success_ratio = cts[True] / (cts[True] + cts[False])
        print("SUCCESS_RATE", success_ratio)
        self.assertGreaterEqual(success_ratio, 0.75)

    def test_get_statistics(self):
        self.zero_init()
        bp = deepcopy(self.bid_ask_prices)
        bp['BTC-USD'] = {'bid': 10000.0, 'ask': 10000.01}
        inv = deepcopy(self.inventory)
        inv['USD'] = 20000.0
        self.broker.initialize(bp, inv)
        
        order = MarketOrder(ccy='BTC-USD', side='long', price=10000.0, size=1.0)

        self.assertTrue(self.broker.portfolio.add_order(order))

        bp['BTC-USD'] = {'bid': 10100.0, 'ask': 10100.01}
        self.broker.step(bp)

        expected_stats = {
            'final_portfolio_value': '$20049.50',
            'initial_portfolio_value': '$20000.00',
            'market_orders': 1,
            'notional_pnl': '+0.248%',
            'realized_pnl': '-50.000%',
        }

        actual_stats = self.broker.get_statistics()
        self.assertDictEqual(expected_stats, actual_stats)


if __name__ == '__main__':
    unittest.main()