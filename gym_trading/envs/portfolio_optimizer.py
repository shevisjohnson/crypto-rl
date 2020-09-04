from abc import ABC, abstractmethod
from collections import deque
from typing import Union, Dict, List
from itertools import filterfalse, permutations

import numpy as np
import pandas as pd
from gym import Env, spaces
from joblib import Parallel, parallel_backend, delayed
from copy import deepcopy
from tqdm import tqdm

from gym_trading.utils.reward import distance_from_optimal_allocation, differential_sharpe_ratio
from configurations import (
    EMA_ALPHA, INDICATOR_WINDOW, INDICATOR_WINDOW_MAX, MARKET_ORDER_FEE, GRIDMAX_LEVEL, LOGGER
)
from gym_trading.utils.meta_broker import MetaBroker
from gym_trading.utils.data_pipeline import DataPipeline
from gym_trading.utils.plot_history import Visualize
from gym_trading.utils.render_env import PortfolioGraph
from gym_trading.utils.penalty_lookup_table import PenaltyLookupTable
from gym_trading.utils.statistic import ExperimentStatistics
from gym_trading.utils.gridmax import gridmax
from indicators import IndicatorManager, RSI, TnS


class PortfolioOptimizer(Env):
    id = 'Portfolio-Optimizer-v0'
    description = "Environment where portfolio allocations are used to create market orders."
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 fiat: str,
                 cryptos: List[str],
                 exchanges: List[str],
                 initial_inventory: Dict[str, np.float32],
                 fitting_file_template: str,
                 testing_file_template: str,
                 window_size: int = 100,
                 seed: int = 1,
                 action_repeats: int = 5,
                 gridmax_level: int = GRIDMAX_LEVEL,
                 training: bool = True,
                 format_3d: bool = False,
                 transaction_fee: bool = True,
                 ema_alpha: list or np.float32 or None = EMA_ALPHA):
        """
        Base class for creating environments extending OpenAI's GYM framework.

        :param symbol: currency pair to trade / experiment
        :param fitting_file: prior trading day (e.g., T-1)
        :param testing_file: current trading day (e.g., T)
        :param max_position: maximum number of positions able to hold in inventory
        :param window_size: number of lags to include in observation space
        :param seed: random seed number
        :param action_repeats: number of steps to take in environment after a given action
        :param training: if TRUE, then randomize starting point in environment
        :param format_3d: if TRUE, reshape observation space from matrix to tensor
        :param reward_type: method for calculating the environment's reward:
            1) 'default' --> inventory count * change in midpoint price returns
            2) 'default_with_fills' --> inventory count * change in midpoint price returns
                + closed trade PnL
            3) 'realized_pnl' --> change in realized pnl between time steps
            4) 'differential_sharpe_ratio' -->
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1.7210&rep=rep1&type=pdf
            5) 'asymmetrical' --> extended version of *default* and enhanced with a
                    reward for being filled above or below midpoint, and returns only
                    negative rewards for Unrealized PnL to discourage long-term
                    speculation.
            6) 'trade_completion' --> reward is generated per trade's round trip


        :param ema_alpha: decay factor for EMA, usually between 0.9 and 0.9999; if NONE,
            raw values are returned in place of smoothed values
        """

        exchanges = sorted(exchanges)

        self.viz = Visualize(
            columns=['total_value', 'allocation', 'pnl', 'realized_pnl'],
            store_historical_observations=True
        )

        # get Broker class to keep track of PnL and orders
        self.broker = MetaBroker(
            fiat=fiat,
            cryptos=cryptos,
            exchanges=exchanges,
            transaction_fee=transaction_fee,
            initial_inventory=initial_inventory
        )

        self.initial_inventory = initial_inventory

        # properties required for instantiation
        self.action_repeats = action_repeats
        self._seed = seed
        self._random_state = np.random.RandomState(seed=self._seed)
        self.training = training
        self.window_size = window_size
        self.reward_type = 'distance_from_optimal_allocation'
        self.format_3d = format_3d  # e.g., [window, features, *NEW_AXIS*]
        self.testing_file_template = testing_file_template

        self.n_currencies = len(self.broker.portfolio.currencies)

        # properties that get reset()
        self.reward = np.float32(0.)
        self.step_reward = np.zeros(1, dtype=np.float32)
        self.A_t = np.zeros(self.n_currencies, dtype=np.float32)
        self.B_t = np.zeros(self.n_currencies, dtype=np.float32)
        self.done = False
        self.local_step_number = 0
        self.midpoints = np.zeros(len(exchanges), dtype=np.float32)
        self.observation = None
        self.action = np.zeros(self.n_currencies, dtype=np.float32)
        self.last_pnl = np.float32(0)
        self.last_midpoints = None
        self.midpoint_changes = None
        self.episode_stats = ExperimentStatistics()
        self.best_bids = {ex: None for ex in exchanges}
        self.best_asks = {ex: None for ex in exchanges}
        self.step_bid_ask_prices = {}

        # get historical data for simulations
        self.data_pipeline = DataPipeline(alpha=ema_alpha)

        # three different data sets, for different purposes:
        #   1) midpoint_prices - midpoint prices that have not been transformed
        #   2) raw_data - raw limit order book data, not including imbalances
        #   3) normalized_data - z-scored limit order book and order flow imbalance
        #       data, also midpoint price feature is replace by midpoint log price change
        self._midpoint_prices, self._raw_data, self._normalized_data = \
            self.data_pipeline.load_portfolio_environment_data(
                exchanges=exchanges,
                fitting_file_template=fitting_file_template,
                testing_file_template=testing_file_template,
                include_imbalances=True,
                as_pandas=True,
            )

        self._midpoint_prices_split = {ex: self._midpoint_prices.filter(regex=f'_{ex}').to_numpy(dtype=np.float32) for ex in exchanges}
        self._raw_data_split = {ex: self._raw_data.filter(regex=f'_{ex}').to_numpy(dtype=np.float32) for ex in exchanges}
        self._normalized_data_split = {ex: self._normalized_data.filter(regex=f'_{ex}').to_numpy(dtype=np.float32) for ex in exchanges}
    
        # derive best bid and offer
        self._best_bids = {ex: self._raw_data[f'midpoint_{ex}'] - (self._raw_data[f'spread_{ex}'] / 2) 
                           for ex in exchanges}
        
        self._best_asks = {ex: self._raw_data[f'midpoint_{ex}'] + (self._raw_data[f'spread_{ex}'] / 2) 
                           for ex in exchanges}

        self.max_steps = self._raw_data.shape[0] - self.action_repeats - 1

        # load indicators into the indicator manager
        self.tns = {}
        self.rsi = {}
        for ex in exchanges:
            self.tns[ex] = IndicatorManager()
            self.rsi[ex] = IndicatorManager()
            for window in INDICATOR_WINDOW:
                self.tns[ex].add(('tns_{}_{}'.format(ex, window), TnS(window=window, alpha=ema_alpha)))
                self.rsi[ex].add(('rsi_{}_{}'.format(ex, window), RSI(window=window, alpha=ema_alpha)))

        # buffer for appending lags
        self.data_buffer = deque(maxlen=self.window_size)

        # Index of specific data points used to generate the observation space
        features = self._raw_data.columns.tolist()
        self.indeces = {ex: {
            'best_bid_index': features.index(f'bids_distance_0_{ex}'),
            'best_ask_index': features.index(f'asks_distance_0_{ex}'),
            'notional_bid_index': features.index(f'bids_notional_0_{ex}'),
            'notional_ask_index': features.index(f'asks_notional_0_{ex}'),
            'buy_trade_index': features.index(f'buys_{ex}'),
            'sell_trade_index': features.index(f'sells_{ex}'),
        } for ex in exchanges}

        self.viz.observation_labels = self._normalized_data.columns.tolist() # normalized stationary L.O.B features
        self.viz.observation_labels += [self.tns[ex].get_labels() + self.rsi[ex].get_labels() for ex in exchanges] # indicators
        self.viz.observation_labels += list(map(lambda x: f'prior_%_{x}', self.broker.portfolio.currencies)) # prior allocation
        self.viz.observation_labels += list(map(lambda x: f'target_%_{x}', self.broker.portfolio.currencies)) # target allocation
        self.viz.observation_labels += ['Realized PNL', 'Notional PNL', 'Reward'] # performance statistics

        # typecast all data sets to numpy
        self._raw_data = self._raw_data.to_numpy(dtype=np.float32)
        self._normalized_data = self._normalized_data.to_numpy(dtype=np.float32)
        self._midpoint_prices = self._midpoint_prices.to_numpy(dtype=np.float32)
        self._best_bids = np.vstack(tuple(self._best_bids[ex].to_numpy(dtype=np.float32) for ex in exchanges)).T
        self._best_asks = np.vstack(tuple(self._best_asks[ex].to_numpy(dtype=np.float32) for ex in exchanges)).T

        # rendering class
        self._render = PortfolioGraph(sym=ex)

        initial_values = np.array([self.broker.portfolio.total_value] * np.shape(self._render.x_vec)[0])

        # graph midpoint prices
        self._render.reset_render_data(
            y_vec=initial_values)

        # Set action and observation spaces
        self.gridmax_level = gridmax_level
        self.actions = gridmax(self.n_currencies, gridmax_level)
        self.plt = PenaltyLookupTable(self.actions)
        self.plt.warm_up()
        self.action_space = spaces.Discrete(len(self.actions))
        # continuous: spaces.Box(low=0.0, high=1.0, shape=(self.n_currencies,), dtype=np.float32)
        self.reset()
        self.observation_space = spaces.Box(low=-10., high=10.,
                                            shape=self.observation.shape,
                                            dtype=np.float32)

    def map_action_to_broker(self, action: np.ndarray) -> (np.float32, np.float32):
        """
        Translate agent's action into an order and submit order to broker.

        :param action: (np.ndarray) agent's target allocation for current step
        :param step_bid_ask_prices: The updated bid and ask prices for the current step
        :return: (tuple) reward, pnl
        """
        target_allocation = action

        step_reward = self._get_step_reward(target_allocation)

        self.broker.reallocate(target_allocation=target_allocation)
        self.broker.step(self.step_bid_ask_prices)

        pnl = np.float32((self.broker.portfolio.total_value / self.broker.portfolio.prior_total_value) - 1.0)

        return step_reward, pnl

    def _get_step_reward(self,
                         target_allocation: np.ndarray) -> np.float32:
        """
        Calculate current step reward using the optimal allocation reward function.

        :param prior_allocation: allocation prior to action
        :param target_allocation: action made by the agent
        :param step_bid_ask_prices: The updated bid and ask prices for the current step
        """
        reward = np.float32(0.)

        prior_allocation = self.broker.portfolio.allocation_arr

        #penalty_table_idx = self.plt.get_table_idx_for_allocation(prior_allocation)

        #reward = np.float32(0)

        for i in range(1, self.n_currencies):
            ex_sym = f'{self.broker.currencies[i]}-{self.broker.fiat}'
            ex_idx = self.broker.exchanges.index(ex_sym)
            tr, self.A_t[i], self.B_t[i] = differential_sharpe_ratio(
                R_t=self.self.midpoint_changes[ex_idx] * self.broker.portfolio.inventory[self.broker.currencies[i]],
                A_tm1=self.A_t[i],
                B_tm1=self.B_t[i]
            )
            reward += tr

        step_penalty = self.plt.get_penalty_for_action(prior_allocation, target_allocation)

        #allocation_deltas = target_allocation * price_deltas_table * penalty

        #pnl = np.sum(allocation_deltas) - np.float32(1)

        #return np.tanh(pnl * np.float32(100))

        return reward + step_penalty

    def step(self, action: int = 0) -> (np.ndarray, np.float32, bool, dict):
        """
        Step through environment with action.

        :param action: (int) action to take in environment
        :return: (tuple) observation, reward, is_done, and empty `dict`
        """
        for current_step in range(self.action_repeats):

            if self.done:
                self.reset()
                return self.observation, self.reward, self.done

            allocation_array = self.broker.portfolio.allocation_arr

            # reset the reward if there ARE action repeats
            if current_step == 0:
                self.reward = np.float32(0.)
                step_action = self.actions[action]
            else:
                step_action = allocation_array

            # Get current step's midpoint and change in midpoint price percentage
            self.midpoints = self._midpoint_prices[self.local_step_number]
            self.midpoint_changes = (self.midpoints / (self.last_midpoints + np.finfo(np.float32).eps))

            # Pass current time step bid/ask prices to broker to calculate PnL,
            # or if any open orders are to be filled
            self.best_bids, self.best_asks = self._get_nbbo()

            # verify the data integrity
            assert all([self.best_bids[ex] <= self.best_asks[ex]
                        for ex in self.broker.exchanges]), (
                "Error: best bid is more expensive than the best Ask:"
                "\nBid = {}\nAsk = {}").format(self.best_bid, self.best_ask)

            self.step_bid_ask_prices = {}
            for i, ex in enumerate(self.broker.exchanges):
                # get buy and sell trade volume to use by indicators and 'broker' to
                # execute any open orders the agent has
                step_buy_volume = self._get_book_data(index=self.indeces[ex]['buy_trade_index'])
                step_sell_volume = self._get_book_data(index=self.indeces[ex]['sell_trade_index'])
                # Update indicators
                self.tns[ex].step(buys=step_buy_volume, sells=step_sell_volume)
                self.rsi[ex].step(price=self.midpoints[i])
                self.step_bid_ask_prices[ex] = {'bid': self.best_bids[ex],
                                          'ask': self.best_asks[ex]}


            # Get PnL from any filled MARKET orders AND action penalties for invalid
            # actions made by the agent for future discouragement
            action_penalty_reward, step_pnl = self.map_action_to_broker(action=step_action)

            self.step_reward = action_penalty_reward

            # Add current step's observation to the data buffer
            step_observation = self._get_step_observation()
            self.data_buffer.append(step_observation)

            # Store for visualization AFTER the episode
            self.viz.add_observation(obs=step_observation)
            self.viz.add(self.broker.portfolio.total_value,  # arguments map to the column names in _init_
                         self.broker.portfolio.allocation,
                         self.broker.portfolio.pnl,
                         self.broker.portfolio.realized_pnl)

            self.reward += self.step_reward
            self.local_step_number += 1
            self.last_midpoints = self.midpoints

        self.observation = self._get_observation()

        if self.local_step_number > self.max_steps:
            self.done = True

            flatten_pnl = self.broker.cash_out()
            self.reward += self.broker.pnl

            # store for visualization after the episode
            self.viz.add(self.broker.portfolio.total_value,  # arguments map to the column names in _init_
                         self.broker.portfolio.allocation,
                         self.broker.portfolio.pnl,
                         self.broker.portfolio.realized_pnl)
        elif self.broker.portfolio.total_value < (0.5 * self.broker.portfolio.initial_total_value):
            self.done = True
            self.reward += self.broker.pnl
            # store for visualization after the episode
            self.viz.add(self.broker.portfolio.total_value,  # arguments map to the column names in _init_
                         self.broker.portfolio.allocation,
                         self.broker.portfolio.pnl,
                         self.broker.portfolio.realized_pnl)


        # save rewards to derive cumulative reward
        self.episode_stats.reward += self.reward

        return self.observation, self.reward, self.done, {}

    def reset(self) -> np.ndarray:
        """
        Reset the environment.

        :return: (np.array) Observation at first step
        """
        if self.training:
            self.local_step_number = self._random_state.randint(low=0,
                                                                high=self.max_steps // 5)
        else:
            self.local_step_number = 0

        # print out episode statistics if there was any activity by the agent
        if self.broker.total_trade_count > 0 or self.broker.realized_pnl != 0.:
            self.episode_stats.number_of_episodes += 1
            print(('-' * 25), '{}-{} {} EPISODE RESET'.format(
                self.broker.exchanges, self._seed, self.reward_type.upper()), ('-' * 25))
            print('Episode Reward: {:.4f}'.format(self.episode_stats.reward))
            print('Episode PnL: {:.2f}%'.format(self.broker.pnl))
            print('Trade Count: {}'.format(self.broker.total_trade_count))
            print('Average PnL per Trade: {:.4f}%'.format(self.broker.pnl / self.broker.total_trade_count))
            print('Total # of episodes: {}'.format(self.episode_stats.number_of_episodes))
            print('\n'.join(['{}\t=\t{}'.format(k, v) for k, v in
                             self.broker.get_statistics().items()]))
            print('First step:\t{}'.format(self.local_step_number))
            print(('=' * 75))
        else:
            print('Resetting environment #{} on episode #{}.'.format(
                self._seed, self.episode_stats.number_of_episodes))

        self.reward = np.float32(0.)
        self.A_t = np.zeros(self.n_currencies, dtype=np.float32)
        self.B_t = np.zeros(self.n_currencies, dtype=np.float32)
        self.done = False
        self.data_buffer.clear()
        self.episode_stats.reset()

        initial_bid_ask_prices = {}

        for i, ex in enumerate(self.broker.exchanges):
            self.rsi[ex].reset()
            self.tns[ex].reset()
            initial_bid_ask_prices[ex] = {'bid': self._best_bids[self.local_step_number][i],
                                          'ask': self._best_asks[self.local_step_number][i]}

        self.broker.reset()
        self.broker.initialize(initial_bid_ask_prices, self.initial_inventory)
        
        self.viz.reset()

        for step in tqdm(range(self.window_size + INDICATOR_WINDOW_MAX + 1)):
            self.midpoints = self._midpoint_prices[self.local_step_number]

            if self.last_midpoints is None:
                self.last_midpoints = self.midpoints

            self.midpoint_changes = (self.midpoints / (self.last_midpoints + np.finfo(np.float32).eps))
            self.best_bids, self.best_asks = self._get_nbbo()
            self.step_bid_ask_prices = {}
            for i, ex in enumerate(self.broker.exchanges):
                step_buy_volume = self._get_book_data(index=self.indeces[ex]['buy_trade_index'])
                step_sell_volume = self._get_book_data(index=self.indeces[ex]['sell_trade_index'])
                self.tns[ex].step(buys=step_buy_volume, sells=step_sell_volume)
                self.rsi[ex].step(price=self.midpoints[i])
                self.step_bid_ask_prices[ex] = {'bid': self.best_bids[ex],
                                          'ask': self.best_asks[ex]}

            self.step_reward = self._get_step_reward(self.broker.portfolio.allocation_arr)
            self.broker.step(self.step_bid_ask_prices)
            # Add current step's observation to the data buffer
            step_observation = self._get_step_observation()
            self.data_buffer.append(step_observation)
            self.local_step_number += 1
            self.last_midpoints = self.midpoints

        self.observation = self._get_observation()

        print('Environment reset complete')

        return self.observation

    def render(self, mode: str = 'human') -> None:
        """
        Render midpoint prices.

        :param mode: (str) flag for type of rendering. Only 'human' supported.
        :return: (void)
        """
        self._render.render(total_value=self.broker.portfolio.total_value,
                            trade_count=self.broker.portfolio.total_trade_count,
                            allocation=self.broker.allocation,
                            mode=mode)

    def close(self) -> None:
        """
        Free clear memory when closing environment.

        :return: (void)
        """
        self.plt.cool_down()
        self.broker.reset()
        self.data_buffer.clear()
        self.episode_stats = None
        self._raw_data = None
        self._normalized_data = None
        self._midpoint_prices = None
        self.tns = None
        self.rsi = None

    def seed(self, seed: int = 1) -> list:
        """
        Set random seed in environment.

        :param seed: (int) random seed number
        :return: (list) seed number in a list
        """
        self._random_state = np.random.RandomState(seed=seed)
        self._seed = seed
        return [seed]

    def _get_nbbo(self) -> (np.float32, np.float32):
        """
        Get best bid and offer.

        :return: (tuple) best bid and offer
        """
        best_bids = {}
        best_asks = {}
        for i, ex in enumerate(self.broker.exchanges):
            best_bids[ex] = self._best_bids[self.local_step_number][i]
            best_asks[ex] = self._best_asks[self.local_step_number][i]
        return best_bids, best_asks

    def _get_book_data(self, index: int = 0) -> np.ndarray or np.float32:
        """
        Return step 'n' of order book snapshot data.

        :param index: (int) step 'n' to look up in order book snapshot history
        :return: (np.array) order book snapshot vector
        """
        return self._raw_data[self.local_step_number][index]

    @staticmethod
    def _process_data(observation: np.ndarray) -> np.ndarray:
        """
        Reshape observation for function approximator.

        :param observation: observation space
        :return: (np.array) clipped observation space
        """
        return np.clip(observation, -10., 10.)

    def _create_prior_allocation_features(self) -> np.ndarray:
        """
        Create agent space feature set reflecting the positions held in inventory.

        :return: (np.array) position features
        """
        return np.array([self.broker.portfolio.prior_allocation[c] 
                         for c in self.broker.portfolio.currencies], dtype=np.float32)

    def  _create_target_allocation_features(self) -> np.ndarray:
        """
        Create a features array for the current time step's action.

        :param action: (int) action number
        :return: (np.array) One-hot of current action
        """
        return np.array([self.broker.portfolio.allocation[c] 
                         for c in self.broker.portfolio.currencies], dtype=np.float32)

    def _create_indicator_features(self) -> np.ndarray:
        """
        Create features vector with environment indicators.

        :return: (np.array) Indicator values for current time step
        """
        return np.array([[self.tns[ex].get_value()] + [self.rsi[ex].get_value()]
                         for ex in self.broker.exchanges], dtype=np.float32).reshape(1, -1).flatten()

    def _create_step_pnl_features(self) -> np.ndarray:
        """
        Create features vector with portfolio step PNL stats.

        :return: (np.array) Indicator values for current time step
        """
        realized_pnl_delta = self.broker.portfolio.realized_pnl - self.broker.portfolio.prior_realized_pnl
        notional_pnl_delta = self.broker.portfolio.pnl - self.broker.portfolio.prior_pnl
        return np.array([realized_pnl_delta, notional_pnl_delta], dtype=np.float32)

    def _get_step_observation(self) -> np.ndarray:
        """
        Current step observation, NOT including historical data.

        :param step_action: (int) current step action
        :return: (np.array) Current step observation
        """
        step_environment_observation = self._normalized_data[self.local_step_number]
        step_indicator_features = self._create_indicator_features()
        step_prior_allocation_features = self._create_prior_allocation_features()
        step_target_allocation_features = self._create_target_allocation_features()
        step_pnl_features = self._create_step_pnl_features()
        observation = np.concatenate((step_environment_observation,
                                      step_indicator_features,
                                      step_prior_allocation_features,
                                      step_target_allocation_features,
                                      step_pnl_features,
                                      np.array([self.step_reward], dtype=np.float32)),
                                     axis=None)
        return self._process_data(observation)

    def _get_observation(self) -> np.ndarray:
        """
        Current step observation, including historical data.

        If format_3d is TRUE: Expand the observation space from 2 to 3 dimensions.
        (note: This is necessary for conv nets in Baselines.)

        :return: (np.array) Observation state for current time step
        """
        # Note: reversing the data to chronological order is actually faster when
        # making an array in Python / Numpy, which is odd. #timeit
        observation = np.asarray(self.data_buffer, dtype=np.float32)
        if self.format_3d:
            observation = np.expand_dims(observation, axis=-1)
        return observation

    def get_trade_history(self) -> pd.DataFrame:
        """
        Get DataFrame with trades from most recent episode.

        :return: midpoint prices, and buy & sell trades
        """
        return self.viz.to_df()

    def plot_trade_history(self, save_filename: Union[str, None] = None) -> None:
        """
        Plot history from back-test with trade executions, total inventory, and PnL.

        :param save_filename: filename for saving the image
        """
        self.viz.plot_episode_history(save_filename=save_filename)

    def plot_observation_history(self, save_filename: Union[str, None] = None) -> None:
        """
        Plot observation space as an image.

        :param save_filename: filename for saving the image
        """
        return self.viz.plot_obs(save_filename=save_filename)
