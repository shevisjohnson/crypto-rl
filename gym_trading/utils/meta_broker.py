# meta_broker.py
#
#   Wrapper class implementing broker.py to manage/monitor a portfolio of
#   multiple currencies and the exchanges between them.
#
#   e.g. 
#       currencies: ['USD', 'BTC', 'ETH']
#       exchanges: ['BTC-USD', 'ETH-BTC', 'ETH-USD']
#
from typing import List, Dict, Optional
from operator import itemgetter

from gym_trading.utils.portfolio import Portfolio
from configurations import LOGGER

class MetaBroker(object):
    def __init__(self,
                 fiat: str = 'USD',
                 cryptos: List[str] = ['ETH', 'BTC'],
                 exchanges: List[str] = ['BTC-USD', 'ETH-USD', 'ETH-BTC'],
                 transaction_fee: bool = True,
                 initial_inventory: Optional[Dict[str, float]] = None,
                 initial_bid_prices: Optional[Dict[str, float]] = None):
        """
        Wrapper around a portfolio which manages order planning/execution
        and monitors trade statistics. Also responsible for retrospectivly
        determining the optimal portfolio allocation for a previous timestep.

        :param fiat: (str)
        :param cryptos: (List[str])
        :param exchanges: (List[str])
        :param transaction_fee: (bool)
        :param initial_inventory: (Optional[Dict[str, float]])
        :param initial_bid_prices: (Optional[Dict[str, float]])
        """
        self.portfolio = Portfolio(fiat=fiat,
                                   cryptos=cryptos,
                                   exchanges=exchanges,
                                   transaction_fee=transaction_fee,
                                   initial_inventory=initial_inventory,
                                   initial_bid_prices=initial_bid_prices)

    def reset(self) -> None:
        """
        Reset broker metrics / portfolio.

        :return: (void)
        """
        self.portfolio.reset()

    def initialize(self, bid_prices: Dict[str, float], inventory: Dict[str, float] = {}) -> None:
        """
        Adds starting values to the portfolio.

        :param inventory: (Dict[str, float]) maps currenies to starting amount
        :param bid_prices: (Dict[str, float]) the most recent bid prices for each
            crypto-fiat exchange. Keys are the crypto symbol, bid_prices[fiat] == 1.0
        """
        self.portfolio.initialize(bid_prices, inventory)

    @property
    def allocation(self) -> Dict[str, float]:
        """
        The fractional breakdown of currencies by fiat value

        :return: (Dict[str, float]) portfolio allocation - sum(allocation.values()) == 1.0
        """
        return self.portfolio.allocation

    def step(self, bid_prices: Dict[str, float]) -> None:
        """
        Step in environment and update portfolio value.

        :param bid_prices: dictionary of the best bid price on each fiat exchange
        :return: (void)
        """
        self.portfolio.step(bid_prices)

    def _validate_allocation(self, allocation: Dict[str, float]) -> None:
        """
        Throws an error if allocation has unknown or missing currencies, or
        if the sum of allocations don't add up to 1.

        :param allocation: (Dict[str, float]) fractional breakdown of currencies by fiat value
        :return: (void)
        """
        if sum(allocation.values()) != 1.0:
            raise ValueError(f"Invalid allocation doesn't add up to 1.0: {allocation}")
        invalid_allocations = [k not in self.portfolio.currencies for k in allocation]
        missing_allocations = [k not in allocation for k in self.portfolio.currencies]
        if any(invalid_allocations):
            raise ValueError("allocation contains unknown currency: " + \
                             f"{list(allocation.keys())[invalid_bid_prices.index(True)]}")
        if any(missing_allocations):
            raise ValueError("Missing allocation for currency: " + \
                             f"{list(self.portfolio.currencies.keys())[missing_bid_prices.index(True)]}")

    def allocate(self, target_allocation: Dict[str, float]) -> bool:
        """
        Generate/execute market orders to shift current allocation to target allocation.

        :param allocation: (Dict[str, float]) fractional breakdown of currencies by fiat value
        :return: (bool) TRUE if all orders were executed successfully, otherwise FALSE
        """
        self._validate_allocation(target_allocation)
        allocation_diffs = {c: target_allocation - self.allocation for c in self.portfolio.currencies}

        max_ingress = max(allocation_diffs.items(), key=operator.itemgetter(1))[0]
        max_egress = min(allocation_diffs.items(), key=operator.itemgetter(1))[0]

        transfer_percent = min(abs(max_ingress), abs(max_egress))

    def get_statistics(self) -> dict:
        """
        Get statistics for long and short inventories.

        :return: statistics
        """
        realized_pnl = self.portfolio.realized_pnl * 100.0
        notional_pnl = self.portfolio.pnl * 100.0
        return dict(
            market_orders=self.portfolio.statistics.market_orders,
            notional_pnl= "{value:+.{precision}f}%".format(value=notional_pnl,precision=3),
            realized_pnl= "{value:+.{precision}f}%".format(value=realized_pnl,precision=3),
            initial_portfolio_value='$%.2f' % self.portfolio.initial_total_value,
            final_portfolio_value='$%.2f' % self.portfolio.total_value
        )