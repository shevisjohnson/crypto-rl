# portfolio.py
#
#   Inventory and risk management module for a collection of currencies
#
#
from collections import deque
from typing import List, Dict, Optional
import numpy as np
from copy import copy

from configurations import LOGGER, SLIPPAGE, MARKET_ORDER_FEE
from gym_trading.utils.order import MarketOrder
from gym_trading.utils.statistic import TradeStatistics


class Portfolio(object):
    def __init__(self, 
                 fiat: str = 'USD',
                 cryptos: List[str] = ['ETH', 'BTC'],
                 exchanges: List[str] = ['BTC-USD', 'ETH-USD', 'ETH-BTC'],
                 transaction_fee: bool = True,
                 initial_inventory: Optional[Dict[str, float]] = None,
                 initial_bid_prices: Optional[Dict[str, float]] = None):
        """
        :param fiat: (str)
        :param cryptos: (List[str])
        :param exchanges: (List[str])
        :param transaction_fee: (bool)
        :param initial_inventory: (Optional[Dict[str, float]])
        :param initial_bid_prices: (Optional[Dict[str, float]])
        """
        self.transaction_fee = transaction_fee
        self.fiat = fiat
        self.cryptos = cryptos
        self.exchanges = exchanges
        self.currencies = [fiat] + cryptos
        self.trades = deque()
        self.statistics = TradeStatistics()
        self.reset()
        if initial_inventory:
            self._initialize_inventory(initial_inventory)
        if initial_bid_prices:
            self._initialize_bid_prices(initial_bid_prices)
        self.initial_total_value = self.total_value
        self.initial_realized_value = self.realized_value

    def reset(self) -> None:
        """
        Reset portfolio metrics / inventories.

        :return: (void)
        """
        self.initial_total_value = 0.0
        self.initial_realized_value = 0.0
        self.trades.clear()
        self.statistics.reset()
        self.inventory = {c: 0.0 for c in self.currencies}
        self.bid_prices = {c: 0.0 for c in self.cryptos}
        self.bid_prices[self.fiat] = 1.0

    def initialize(self, bid_prices: Dict[str, float], inventory: Dict[str, float] = {}) -> None:
        """
        Adds starting values to the portfolio.

        :param inventory: (Dict[str, float]) maps currenies to starting amount
        :param bid_prices: (Dict[str, float]) the most recent bid prices for each
            crypto-fiat exchange. Keys are the crypto symbol, bid_prices[fiat] == 1.0
        """
        self._initialize_bid_prices(bid_prices)
        self._initialize_inventory(inventory)
        self.initial_total_value = self.total_value
        self.initial_realized_value = self.realized_value

    def __str__(self):
        msg = f'Portfolio: [allocation={self.allocation} | pnl={self.pnl}'
        msg += f' | realized_pnl={self.realized_pnl}'
        msg += f' | total_value={self.total_value}'
        msg += f' | realized_value={self.realized_value}'
        msg += f' | total_trade_count={self.total_trade_count}]'
        return msg

    @property
    def value_breakdown(self) -> Dict[str, float]:
        """
        Returns a dictionary of each position mapped to it's 
        current value in fiat currency.

        :return: (Dict[str, float]) portfolio value breakdown
        """
        return {
            currency: (count * self.bid_prices[currency])
            for currency, count in self.inventory.items()
        }

    @property
    def unrealized_value(self) -> float:
        """
        calculates the unrealized value of portfolio in fiat currency

        :return: (float) portfolio value (e.g. USD)
        """
        return sum([self.value_breakdown[c] for c in self.cryptos])

    @property
    def realized_value(self) -> float:
        """
        :return: (float) value of fiat holdings in portfolio
        """
        return copy(self.inventory[self.fiat])

    @property
    def total_value(self) -> float:
        """
        calculates the total value of portfolio in fiat currency

        :return: (float) portfolio value (e.g. USD)
        """
        return sum(list(self.value_breakdown.values()))

    @property
    def allocation(self) -> Dict[str, float]:
        """
        The fractional breakdown of currencies by fiat value

        :return: (Dict[str, float]) portfolio allocation - sum(allocation.values()) == 1.0
        """
        if self.total_value <= 0.0:
            return {c: 0.0 for c in self.currencies}
        return {c: (v / self.total_value) for c, v in self.value_breakdown.items()}

    @property
    def total_trade_count(self) -> int:
        return self.trades.__len__()

    @property
    def pnl(self) -> float:
        if self.initial_total_value != 0.0:
            return (self.total_value / self.initial_total_value) - 1.0
        return 0.0

    @property
    def realized_pnl(self) -> float:
        if self.initial_realized_value != 0.0:
            return (self.realized_value / self.initial_realized_value) - 1.0
        return 0.0

    def _validate_inventory(self, inventory: Dict[str, float]) -> None:
        invalid_inventory_items = [k not in self.currencies for k in inventory]
        if any(invalid_inventory_items):
            raise ValueError("inventory contains unknown currency: " + \
                             f"{list(inventory.keys())[invalid_inventory_items.index(True)]}")

    def _initialize_inventory(self, inventory: Dict[str, float] = {}) -> None:
        self._validate_inventory(inventory)
        self.inventory.update(inventory)

    def _validate_bid_prices(self, bid_prices: Dict[str, float]) -> None:
        invalid_bid_prices = [k not in self.currencies for k in bid_prices]
        missing_bid_prices = [k not in bid_prices for k in self.currencies]
        if any(invalid_bid_prices):
            raise ValueError("bid_prices contains unknown currency: " + \
                             f"{list(bid_prices.keys())[invalid_bid_prices.index(True)]}")
        if any(missing_bid_prices):
            raise ValueError("Missing bid price for currency: " + \
                             f"{list(self.currencies.keys())[missing_bid_prices.index(True)]}")

    def _initialize_bid_prices(self, bid_prices: Dict[str, float]) -> None:
        self._validate_bid_prices(bid_prices)
        self.bid_prices.update(bid_prices)
        self.bid_prices[self.fiat] = 1.0

    def step(self, bid_prices: Dict[str, float]) -> None:
        """
        Step in environment and update portfolio value.

        :param bid_prices: dictionary of the best bid price on each fiat exchange
        :return: (void)
        """
        self._validate_bid_prices(bid_prices)
        self.bid_prices.update(bid_prices)
        self.bid_prices[self.fiat] = 1.0

    def add_order(self, order: MarketOrder) -> bool:
        """
        Add a MARKET order.

        :param order: (Order) New order to be used for updating existing order or
                        placing a new order
        :return: (bool) TRUE if trade was successfully executed, FALSE otherwise.
        """
        if order.ccy not in self.exchanges:
            LOGGER.info(f"Invalid order on unknown exchange: {order.ccy}")
            return False

        asset, base = order.ccy.split('-')

        buying_asset = bool(['short', 'long'].index(order.side))

        # Create a hypothetical average execution price incorporating a fixed slippage
        if buying_asset:
            order.average_execution_price = order.price * (1.0 + SLIPPAGE)
        else:
            order.average_execution_price = order.price * (1.0 - SLIPPAGE)

        # Determine impact of order on inventory
        # ingress is where value is going
        ingress_sym = [asset, base][not int(buying_asset)]
        ingress_price = order.average_execution_price \
                     if not buying_asset \
                     else 1.0
        ingress_amount = order.size * ingress_price
        if self.transaction_fee:
             ingress_amount *= 1.0 - MARKET_ORDER_FEE

        # egress is where value is coming from
        egress_sym = [asset, base][int(buying_asset)]
        egress_price = order.price \
                      if buying_asset \
                      else 1.0
        egress_amount = order.size * egress_price

        if self.inventory[egress_sym] < egress_amount:
            LOGGER.info(f"Invalid order, too large. [ccy={order.ccy}, cost={egress_amount}, available={self.inventory[egress_sym]}]")
            return False

        # Update portfolio inventory attributes
        self.inventory[ingress_sym] += ingress_amount
        self.inventory[egress_sym] -= egress_amount

        # execute and save the market order
        order.executed = order.size
        self.trades.append(order)

        # update statistics
        self.statistics.market_orders += 1

        LOGGER.debug(
            '  %s @ %.2f | step %i' % (
                order.side, order.average_execution_price, order.step)
        )
        return True

    def generate_exchange_graph(self) -> Dict[str, Dict[str, str]]:
        """
        Creates an undirected graph from list of currencies and exchanges. 
        Currencies and exchanges are represented as vertices and edges respectively.

        The graph is represented as nested dicts. Each key of each dict is a vertex,
        each leaf node is the exhange symbol for the vertices leading to that leaf.

        example:
        self.currencies = ['USD', 'ETH', 'BTC']
        self.exchanges = ['BTC-USD', 'ETH-USD', 'ETH-BTC']
        
        graph = {
            'USD': {
                'BTC': 'BTC-USD',
                'ETH': 'ETH-USD',
            }
            'BTC': {
                'USD': 'BTC-USD',
                'ETH': 'ETH-BTC',
            }
            'ETH': {
                'BTC': 'ETH-BTC',
                'USD': 'ETH-USD',
            }
        }

        :return: (Dict[str, Dict[str, str]]) currency exchange graph
        """
        graph = {}
        for vertex in self.currencies:
            edges = {}
            for edge in self.exchanges:
                edge_ends = edge.split('-')
                if vertex in edge_ends:
                    idx = int(not bool(edge_ends.index(vertex)))
                    edges[edge_ends[idx]] = edge
            graph[vertex] = edges
        return graph