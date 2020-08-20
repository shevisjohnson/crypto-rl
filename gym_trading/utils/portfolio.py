# portfolio.py
#
#   Inventory and risk management module for a collection of positions
#
#
from collections import deque
from typing import List
import numpy as np

from configurations import LOGGER, SLIPPAGE, MARKET_ORDER_FEE
from gym_trading.utils.order import MarketOrder
from gym_trading.utils.statistic import TradeStatistics

def generate_exchange_graph(currencies: List[str],
                            exchanges: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Creates an undirected graph from list of currencies and exchanges. 
    Currencies and exchanges are represented as vertices and edges respectively.

    The graph is represented as nested dicts. Each key of each dict is a vertex,
    each leaf node is the exhange symbol for the vertices leading to that leaf.

    example:
    currencies = ['USD', 'ETH', 'BTC']
    exchanges = ['BTC-USD', 'ETH-USD', 'ETH-BTC']
     
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
    for vertex in currencies:
        edges = {}
        for edge in exchanges:
            edge_ends = edge.split('-')
            if vertex in edge_ends:
                idx = int(not bool(edge_ends.index(vertex)))
                edges[edge_ends[idx]] = edge
        graph[vertex] = edges
    return graph

class Portfolio(object):
    def __init__(self, 
                 fiat: str = 'USD',
                 cryptos: List[str] = ['ETH', 'BTC'],
                 exchanges: List[str] = ['BTC-USD', 'ETH-USD', 'ETH-BTC'],
                 transaction_fee: bool = False):

        self.transaction_fee = transaction_fee
        self.fiat = fiat
        self.cryptos = cryptos
        self.exchanges = exchanges
        self.currencies = [fiat] + cryptos
        self.trades = deque()
        self.inventory = {c: 0.0 for c in self.currencies}
        self.bid_prices = {c: 0.0 for c in cryptos}
        self.bid_prices[fiat] = 1.0 # fiat converts to itself 1:1
        self.realized_pnl = 0.0
        self.previous_fiat_value = 0.0
        self.exchange_graph = generate_exchange_graph(self.currencies, self.exchanges)
        self.statistics = TradeStatistics()

    @property
    def fiat_value(self) -> float:
        """
        calculate the total value of portfolio in fiat currency

        :return: (float) portfolio value (e.g. USD)
        """
        total = 0.0
        for c in self.currencies:
            total += self.bid_prices[c] * self.inventory[c]
        return total

    @property
    def allocation(self) -> Dict[str, float]:
        """
        The fractional breakdown of currencies by fiat value

        :return: (Dict[str, float]) portfolio allocation - sum(allocation.values()) == 1.0
        """
        fiat_value = self.fiat_value

        allocation = {}

        for currency, count in self.inventory.items():
            allocation[currency] = (count * self.bid_prices[currency]) / fiat_value
        
        return allocation

    @property
    def total_trade_count(self):
        return self.trades.__len__()

    def __str__(self):
        msg = f'Portfolio: [allocation={self.allocation} | realized_pnl={self.realized_pnl}'
        msg += f' | fiat_value={self.fiat_value}'
        msg += f' | total_trade_count={self.total_trade_count}]'
        return msg

    def reset(self) -> None:
        """
        Reset broker metrics / inventories.

        :return: (void)
        """
        self.realized_pnl = 0.0
        self.trades.clear()
        self.statistics.reset()
        self.currency_counter = {c: 0.0 for c in self.currencies}
        self.bid_prices = {c: 0.0 for c in self.cryptos}
        self.bid_prices[self.fiat] = 1.0

    def _validate_inventory(self, inventory: Dict[str, float]) -> None:
        invalid_inventory_items = [k not in self.currencies for k in inventory]
        if any(invalid_inventory_items):
            raise ValueError("inventory contains unknown currency: " + \
                             f"{list(inventory.keys())[invalid_inventory_items.index(True)]}")

    def _validate_bid_prices(self, bid_prices: Dict[str, float]) -> None:
        invalid_bid_prices = [k not in self.currencies for k in bid_prices]
        missing_bid_prices = [k not in bid_prices for k in self.currencies]
        if any(invalid_bid_prices):
            raise ValueError("bid_prices contains unknown currency: " + \
                             f"{list(bid_prices.keys())[invalid_bid_prices.index(True)]}")
        if any(missing_bid_prices):
            raise ValueError("Missing bid price for currency: " + \
                             f"{list(self.currencies.keys())[missing_bid_prices.index(True)]}")


    def initialize(self, inventory: Dict[str, float], bid_prices: Dict[str, float]) -> None:
        self._validate_inventory(inventory)
        self._validate_bid_prices(bid_prices)
        self.inventory.update(inventory)
        self.bid_prices.update(bid_prices)
        self.bid_prices[self.fiat] = 1.0
        self.previous_fiat_value = self.fiat_value

    def step(self, bid_prices: Dict[str, float]) -> None:
        """
        Step in environment and update portfolio value.

        :param bid_prices: dictionary of the best bid price on each fiat exchange
        :return: (void)
        """
        self._validate_bid_prices(bid_prices)
        self.bid_prices.update(bid_prices)
        self.bid_prices[self.fiat] = 1.0

        self.realized_pnl += (self.fiat_value / self.previous_fiat_value) - 1
        self.previous_fiat_value = self.fiat_value

    def add_order(self, order: MarketOrder) -> bool:
        """
        Add a MARKET order.

        :param order: (Order) New order to be used for updating existing order or
                        placing a new order
        :return: (bool) TRUE if trade was successfully executed, FALSE otherwise.
        """
        if order.ccy not in self.exchanges:
            LOGGER.debug(f"Invalid order on unknown exchange: {order.ccy}")
            return False

        asset, base = order.ccy.split('-')

        buying_asset = bool(['short', 'long'].index(order.side))

        buy_sym = [asset, base][not int(buying_asset)]
        buy_price = order.price if not buying_asset else 1.0 / order.price

        sell_sym = [asset, base][int(buying_asset)]
        sell_price = order.price if buying_asset else 1.0 / order.price

        if self.inventory[sell_sym] < (sell_price * order.size):
            LOGGER.debug(f"Invalid order, too large. [ccy={order.ccy}, cost={sell_price * order.size}, available={self.inventory[selling]}]")
            return False

        # Create a hypothetical average execution price incorporating a fixed slippage
        #average_buy_price = buy_price * (1.0 + SLIPPAGE)
        #average_sell_price = sell_price * (1.0 - SLIPPAGE)

        order.average_execution_price = order.price
        order.executed = order.size

        # Update portfolio inventory attributes
        self.trades.append(order)  # execute and save the market order
        self.inventory[buy_sym] += (order.size * buy_price) * (1.0 - MARKET_ORDER_FEE)
        self.inventory[sell_sym] -= order.size * sell_price

        # update statistics
        self.statistics.market_orders += 1

        LOGGER.debug(
            '  %s @ %.2f | step %i' % (
                order.side, order.average_execution_price, order.step)
        )
        return True