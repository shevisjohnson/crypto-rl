import argparse
from typing import Union, Dict, List
from agent.dqn import Agent
from configurations import LOGGER, FIAT
from configurations import CRYPTOS, EXCHANGES, INITIAL_ALLOCATION, INITIAL_INVENTORY
from numpy import float32
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--window_size',
                    default=100,
                    help="Number of lags to include in the observation",
                    type=int)
parser.add_argument('--initial_inventory',
                    default=INITIAL_INVENTORY,
                    help="Maximum number of positions that are " +
                         "able to be held in a broker's inventory",
                    type=int)
parser.add_argument('--fitting_file_template',
                    default='demo_{}_2020-08-09.csv.xz',
                    help="Data set for fitting the z-score scaler (previous day)",
                    type=str)
parser.add_argument('--testing_file_template',
                    default='demo_{}_2020-08-10.csv.xz',
                    help="Data set for training the agent (current day)",
                    type=str)
parser.add_argument('--fiat',
                    default=FIAT,
                    help="Name of currency pair or instrument",
                    type=str)
parser.add_argument('--cryptos',
                    default=CRYPTOS,
                    help="Name of currency pair or instrument",
                    type=str) 
parser.add_argument('--initial_inventory',
                    default=INITIAL_INVENTORY,
                    help="Name of currency pair or instrument",
                    type=Dict[str, float32])

parser.add_argument('--number_of_training_steps',
                    default=1e5,
                    help="Number of steps to train the agent "
                         "(does not include action repeats)",
                    type=int)
parser.add_argument('--gamma',
                    default=0.99,
                    help="Discount for future rewards",
                    type=float)
parser.add_argument('--seed',
                    default=1,
                    help="Random number seed for data set",
                    type=int)
parser.add_argument('--action_repeats',
                    default=5,
                    help="Number of steps to pass on between actions",
                    type=int)
parser.add_argument('--load_weights',
                    default=False,
                    help="Load saved load_weights if TRUE, otherwise start from scratch",
                    type=bool)
parser.add_argument('--visualize',
                    default=False,
                    help="Render midpoint on a screen",
                    type=bool)
parser.add_argument('--training',
                    default=True,
                    help="Training or testing mode. " +
                         "If TRUE, then agent starts learning, " +
                         "If FALSE, then agent is tested",
                    type=bool)
parser.add_argument('--nn_type',
                    default='cnn',
                    help="Type of neural network to use: 'cnn' or 'mlp' ",
                    type=str)
parser.add_argument('--dueling_network',
                    default=True,
                    help="If TRUE, use Dueling architecture in DQN",
                    type=bool)
parser.add_argument('--double_dqn',
                    default=True,
                    help="If TRUE, use double DQN for Q-value estimation",
                    type=bool)
args = vars(parser.parse_args())


def main(kwargs):
    LOGGER.info(f'Experiment creating agent with kwargs: {kwargs}')
    agent = Agent(**kwargs)
    LOGGER.info(f'Agent created. {agent}')
    agent.start()


def train(kwargs):
     dates = ['2020-08-09', '2020-08-09', '2020-08-09']
     for i, d in enumerate(dates):
          twargs = deepcopy(kwargs)
          twargs['fitting_file'] = 'test'
          agent = Agent(**twargs)


if __name__ == '__main__':
    main(kwargs=args)
