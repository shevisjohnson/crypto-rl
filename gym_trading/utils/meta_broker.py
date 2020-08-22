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
from collections import OrderedDict
import numpy as np

from gym_trading.utils.portfolio import Portfolio
from gym_trading.utils.order import MarketOrder
from gym_trading.utils.exchange_graph import generate_exchange_graph
from configurations import LOGGER, MAX_TRADES_PER_ACTION, ALLOCATION_TOLERANCE

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
        and monitors trade statistics.

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
        self.exchange_graph = generate_exchange_graph(self.portfolio.exchanges)

    def reset(self) -> None:
        """
        Reset broker metrics / portfolio.

        :return: (void)
        """
        self.exchange_graph = generate_exchange_graph(self.portfolio.exchanges)
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
        for ccy, prices in bid_ask_prices.items():
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
        if not np.isclose(np.float64(np.sum(np.array(list(allocation.values())))), np.float64(1.0)):
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
        Generate and execute market orders to shift from current allocation to target allocation.

        :param target_allocation: (Dict[str, float]) fractional breakdown of currencies by fiat value
        :return: (bool) TRUE if target allocation reached, otherwise FALSE
        """
        self._validate_allocation(target_allocation)

        # Short circuit if we have already reached our target
        if self._close_to(target_allocation):
            return True

        for _ in range(MAX_TRADES_PER_ACTION):
            # get difference between current allocation and target
            allocation_diffs = {c: target_allocation[c] - self.allocation[c] 
                                for c in self.portfolio.currencies}

            # Identify positive and negative differences with maximum magnitude
            sorted_diffs = sorted(list(allocation_diffs.items()), key=itemgetter(1))

            ingress_idx = -1
            egress_idx = 0

            max_ingress = sorted_diffs[ingress_idx]
            max_egress = sorted_diffs[egress_idx]

            # If no available exchange, find a middleman currency
            if max_egress[0] not in self.exchange_graph[max_ingress[0]]:
                for i in range(1, len(self.portfolio.currencies)):
                    new_ingress = sorted_diffs[ingress_idx - i]
                    if max_egress[0] in self.exchange_graph[new_ingress[0]] and \
                       max_ingress[0] in self.exchange_graph[new_ingress[0]]:
                        max_ingress = new_ingress
                        break

            # Generate order details
            exchange_data = self.exchange_graph[max_ingress[0]][max_egress[0]]
            asset, base = exchange_data['ccy'].split('-')
            buying_asset = max_ingress[0] == asset
            order_side = ['short', 'long'][int(buying_asset)]
            price_basis = ['bid', 'ask'][int(buying_asset)]
            order_price = exchange_data[price_basis]
            transfer_percent = abs(max_egress[1])
            transfer_value_in_fiat = transfer_percent * self.portfolio.total_value
            transfer_value_in_base = transfer_value_in_fiat / self.portfolio.bid_prices[base]
            order_size = transfer_value_in_base / order_price

            LOGGER.debug("exchange_data", exchange_data)
            LOGGER.debug("order_side", order_side)
            LOGGER.debug("order_price", order_price)
            LOGGER.debug("transfer_percent", transfer_percent)
            LOGGER.debug("transfer_value_in_fiat", transfer_value_in_fiat)
            LOGGER.debug("transfer_value_in_base", transfer_value_in_base)
            LOGGER.debug("order_size", order_size)

            LOGGER.debug("old allocation", self.portfolio.allocation)

            # Create and execute order
            order = MarketOrder(ccy=exchange_data['ccy'],
                                side=order_side,
                                price=order_price,
                                size=order_size)
            self.portfolio.add_order(order)

            LOGGER.debug("new allocation", self.portfolio.allocation)
            LOGGER.debug("-----------------------------")

            # Check whether we have reached our target
            if self._close_to(target_allocation):
                return True
        # Reached maximum number of trades
        return False

    def _close_to(self, target_allocation: Dict[str, float]) -> bool:
        if all([np.isclose(self.allocation[c], target_allocation[c], atol=ALLOCATION_TOLERANCE) \
                for c in self.portfolio.currencies]):
            return True
        return False

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