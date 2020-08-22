import unittest
import numpy as np
from scipy.special import softmax
from copy import deepcopy
from pprint import pprint, pformat

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
        currency_amounts = np.random.choice(10000,len(self.broker.portfolio.currencies)) / np.float64(10)
        exchange_rates = np.random.choice(10000,len(self.broker.portfolio.exchanges)) / np.float64(10.0)
        self.final_init(currency_amounts, exchange_rates)

    def zero_init(self):
        currency_amounts = np.zeros(len(self.broker.portfolio.currencies))
        exchange_rates = np.zeros(len(self.broker.portfolio.exchanges))
        self.final_init(currency_amounts, exchange_rates)

    def final_init(self, currency_amounts, exchange_rates):
        self.inventory = dict(zip(self.broker.portfolio.currencies, currency_amounts))
        self.bid_ask_prices = dict(zip(self.broker.portfolio.exchanges,
                                  [{'ask': np.float64(v) * np.float64(1.00001), 'bid': np.float64(v) * np.float64(0.99999)} for v in exchange_rates]))
        self.broker.initialize(self.bid_ask_prices, self.inventory)

    def test_step_updates_portfolio_attributes(self):
        self.random_init()
        total_value_before_step = self.broker.portfolio.total_value
        pnl_before_step = self.broker.portfolio.pnl
        for i in range(5):
            vals2 = np.random.choice(10000,len(self.broker.portfolio.currencies)) / np.float64(10.0)
            bid_prices = dict(zip(self.broker.portfolio.currencies, vals2))
            self.broker.portfolio.step(bid_prices)
        total_value_after_step = self.broker.portfolio.total_value
        pnl_after_step = self.broker.portfolio.pnl
        self.assertNotEqual(total_value_before_step, total_value_after_step)
        self.assertNotEqual(pnl_before_step, pnl_after_step)

    def test_add_order_updates_portfolio_attributes(self):
        self.zero_init()
        bp = deepcopy(self.bid_ask_prices)
        bp['BTC-USD'] = {'bid': np.float64(11000.0), 'ask': np.float64(11000.01)}
        inv = deepcopy(self.inventory)
        inv['USD'] = np.float64(11000.0)
        self.broker.initialize(bp, inv)

        total_value_before_order = self.broker.portfolio.total_value
        pnl_before_order = self.broker.portfolio.pnl
        realized_value_before_order = self.broker.portfolio.realized_value
        realized_pnl_before_order = self.broker.portfolio.realized_pnl
        
        order = MarketOrder(ccy='BTC-USD', side='long', price=np.float64(11000.0), size=np.float64(1.0))

        self.assertTrue(self.broker.portfolio.add_order(order))

        total_value_after_order = self.broker.portfolio.total_value
        pnl_after_order = self.broker.portfolio.pnl
        realized_value_after_order = self.broker.portfolio.realized_value
        realized_pnl_after_order = self.broker.portfolio.realized_pnl

        self.assertNotEqual(total_value_before_order, total_value_after_order)
        self.assertNotEqual(pnl_before_order, pnl_after_order)
        self.assertNotEqual(realized_value_before_order, realized_value_after_order)
        self.assertNotEqual(realized_pnl_before_order, realized_pnl_after_order)

    def test_reallocation(self):
        exchange_rates = [
            np.float64(11812.31),
            np.float64(411.30),
            np.float64(15.22256),
            np.float64(0.105917),
            np.float64(0.03478),
            np.float64(0.00000898),
            np.float64(0.03709380),
        ]
        bid_ask_prices = dict(zip(EXCHANGES,
                                  [{'ask': v * np.float64(1.00001), 'bid': v * np.float64(0.99999)} for v in exchange_rates]))
        inventory = {}
        for c in self.broker.portfolio.currencies:
            if c == 'USD':
                inventory[c] = np.float64(10000.0)
            else:
                inventory[c] = np.float64(10000.0) / bid_ask_prices[f'{c}-USD']['bid']

        self.broker.initialize(bid_ask_prices, inventory)

        cts = {True: np.float64(0.0), False: np.float64(0.0)}
        for _ in range(1000):
            initial_allocation = self.broker.allocation

            currency_range = range(len(self.broker.portfolio.currencies))
            
            nums = np.array([np.float64(np.random.uniform()) for _ in currency_range])
            total = sum(nums)

            prior_allocation = self.broker.allocation

            target_allocation = {self.broker.portfolio.currencies[i]: nums[i] / total
                                for i in currency_range}
            prior_trades = self.broker.portfolio.total_trade_count
            reached_target = self.broker.reallocate(target_allocation)
            if not reached_target:
                allocation_diffs = [target_allocation[c] - self.broker.allocation[c] 
                                    for c in self.broker.portfolio.currencies]
                logs = {
                    "BEFORE": prior_allocation,
                    "TARGET": target_allocation,
                    "AFTER": self.broker.allocation,
                    "DIFF": allocation_diffs,
                    "TRADES": self.broker.portfolio.total_trade_count - prior_trades,
                }
                self.assertLessEqual(max(allocation_diffs), np.float64(0.01), f"should never miss target by more than 1%.\n{pformat(logs)}")
            cts[reached_target] += np.float64(1.0)

        success_rate = cts[True] / (cts[True] + cts[False])
        #print("SUCCESS_RATE", success_rate)
        self.assertGreaterEqual(success_rate, np.float64(0.75), 'should hit target more than 75% of the time')

    def test_get_statistics(self):
        self.zero_init()
        bp = deepcopy(self.bid_ask_prices)
        bp['BTC-USD'] = {'bid': np.float64(10000.0), 'ask': np.float64(10000.01)}
        inv = deepcopy(self.inventory)
        inv['USD'] = np.float64(20000.0)
        self.broker.initialize(bp, inv)
        
        order = MarketOrder(ccy='BTC-USD', side='long', price=np.float64(10000.0), size=np.float64(1.0))

        self.assertTrue(self.broker.portfolio.add_order(order))

        bp['BTC-USD'] = {'bid': np.float64(10100.0), 'ask': np.float64(10100.01)}
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