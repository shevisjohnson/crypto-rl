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
from gym_trading.utils.order import MarketOrder
from gym_trading.utils.exchange_graph import generate_exchange_graph
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
        self.exchange_graph = generate_exchange_graph(self.portfolio.currencies)

    def reset(self) -> None:
        """
        Reset broker metrics / portfolio.

        :return: (void)
        """
        self.exchange_graph = generate_exchange_graph(self.portfolio.currencies)
        self.portfolio.reset()

    def initialize(self, 
                   bid_ask_prices: Dict[str, Dict[str, float]], 
                   inventory: Dict[str, float] = {}) -> None:
        """
        Adds starting values to the portfolio and exchagne graph.

        :param bid_ask_prices: (Dict[str, Dict[str, float]]) the most recent bid/ask prices for each
            exchange. Keys are the exchange symbol
        :param inventory: (Dict[str, float]) maps currenies to starting amount
        :return: (void)
        """
        self._validate_bid_ask_prices(bid_ask_prices)
        self._update_exchange_graph(bid_ask_prices)
        bid_prices = {}
        for c in self.portfolio.cryptos:
            bid_prices[c] = self.exchange_graph[c][self.portfolio.fiat]['bid']
        bid_prices[self.portfolio.fiat] = 1.0
        self.portfolio.initialize(bid_prices, inventory)

    @property
    def allocation(self) -> Dict[str, float]:
        """
        The fractional breakdown of currencies by fiat value

        :return: (Dict[str, float]) portfolio allocation - sum(allocation.values()) == 1.0
        """
        return self.portfolio.allocation
        
    def _update_exchange_graph(self, bid_ask_prices: Dict[str, Dict[str, float]]) -> None:
        for ccy, prices in bid_ask_prices.items():
            asset, base = ccy.split('-')
            self.exchange_graph[asset][base].update(prices)
            self.exchange_graph[base][asset].update(prices)

    def _validate_bid_ask_prices(self, bid_ask_prices: Dict[str, Dict[str, float]]) -> None:
        for ccy, prices in bid_ask_prices:
            if ccy not in self.portfolio.exchanges:
                raise ValueError(f"bid_ask_prices contains unknown exchange: {ccy}")
            elif 'ask' not in prices:
                raise ValueError(f"missing ask data for exchange: {ccy}")
            elif 'bid' not in prices:
                raise ValueError(f"missing bid data for exchange: {ccy}")
        for ccy in self.portfolio.exchanges:
            if ccy not in bid_ask_prices:
                raise ValueError(f"missing price data for exchange: {ccy}")

    def step(self, bid_ask_prices: Dict[str, Dict[str, float]]) -> None:
        """
        Step in environment and update portfolio values.

        :param bid_ask_prices: dictionary of the best bid and ask price on each exchange
        :return: (void)
        """
        self._validate_bid_ask_prices(bid_ask_prices)
        self._update_exchange_graph(bid_ask_prices)
        bid_prices = {}
        for c in self.portfolio.cryptos:
            bid_prices[c] = self.exchange_graph[c][self.portfolio.fiat]['bid']
        bid_prices[self.portfolio.fiat] = 1.0
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

    def reallocate(self, target_allocation: Dict[str, float]) -> bool:
        """
        Generate/execute market orders to shift current allocation to target allocation.

        :param allocation: (Dict[str, float]) fractional breakdown of currencies by fiat value
        :return: (bool) TRUE if target allocation reached, otherwise FALSE
        """
        self._validate_allocation(target_allocation)

        # Check if we already reached target
        if all([self.allocation[c] == target_allocation[c] \
                for c in self.portfolio.currencies]):
            return True

        # get difference between current allocation and target
        allocation_diffs = {c: target_allocation - self.allocation for c in self.portfolio.currencies}

        # Identify largest positive and negative differences
        max_ingress = max(allocation_diffs.items(), key=itemgetter(1))
        max_egress = min(allocation_diffs.items(), key=itemgetter(1))

        exchange_data = self.exchange_graph[max_ingress[0]][max_egress[0]]

        asset, base = exchange_data['ccy'].splt('-')

        buying_asset = max_egress == asset

        order_side = ['short', 'long'][int(buying_asset)]
        price_basis = ['bid', 'ask'][int(buying_asset)]

        order_price = exchange_data[price_basis]

        transfer_percent = min(abs(max_ingress[1]), abs(max_egress[1]))

        transfer_value_in_fiat = transfer_percent * self.portfolio.total_value

        order_size = transfer_value_in_fiat / self.portfolio.bid_prices[base]

        order = MarketOrder(ccy=exchange_data['ccy'],
                            side=order_side,
                            price=order_price,
                            size=order_size)

        self.portfolio.add_order(order)


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