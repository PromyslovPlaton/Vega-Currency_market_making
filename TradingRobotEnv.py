import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import math
import numpy as np

import tensorflow as tf

from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent

#from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step
from tf_agents.utils import common

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import PIL.ImageDraw
import PIL.Image
from PIL import ImageFont

import Iterator

start_capital = 1e9 # стартовый капитал
const_k = 0.7 # константа из функции вероятности исполнения, k > 0
const_disc = 1/1.03 # показатель инфляции \beta
const_r = 0.3 # константа для риска, r > 0
currency_pair = ['BTC/USD'] # рассматриваемая валютная пара
order_size = 1000 # размер заявки, который мы можем выставить

class TradingRobotEnv(gym.Env):
    metadata = {
        'render.modes': ['robot', 'rgb_array'],
        'video.frames_per_second': 1
    }

    STATE_ELEMENTS = 5
    STATES = [
        'our_usd', 'our_btc',
        'market_bid', 'market_ask',
        'net_worth'
    ]
    STATE_OUR_USD = 0
    STATE_OUR_BTC = 1
    STATE_MARKET_BID = 2
    STATE_MARKET_ASK = 3
    STATE_NET_WORTH = 4

    ACTION_ELEMENTS = 4
    ACTION_BID = 0
    ACTION_ASK = 1
    ACTION_BID_LEVEL = 2
    ACTION_ASK_LEVEL = 3

    COEFFICIENT_K = const_k
    DISCOUNTING_FACTOR = const_disc
    DERIVATIVE_OF_RISK = const_r
    STARTING_CAPITAL = start_capital
    CURRENCY_PAIR = currency_pair
    ORDER_SIZE = order_size

    def __init__(self, goal_velocity=0):
        # если True то вывод подробней для отладки
        # self.verbose = False
        self.verbose = True
        self.viewer = None
        self.time = total_time

        self.action_space = Dict({'Bid_Ask': MultiBinary(2), 'Level_BA': Box(low_level, high_level, shape=(2,))})
        self.observation_space = spaces.Box(
            low=0.0,
            high=1e12,
            shape=(TradingRobotEnv.STATE_ELEMENTS,),
            dtype=np.float32
        )

        self.reset()

        self.state_log = []
        self.state_ProfitTime = []

        self.seed()
        self.reset()

        # пустой журнал нужен для отладки,
        # действительно ли наша модель делает то
        self.state_log = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # функция подсчета капитала
    def _calc_net_worth(self):
        # Кол-во валюты в корзине
        num_usd = float(self.state[TradingRobotEnv.STATE_OUR_USD])
        num_btc = float(self.state[TradingRobotEnv.STATE_OUR_BTC])
        # Курс обмена
        course = (float(self.state[TradingRobotEnv.STATE_MARKET_ASK]) + \
                 float(self.state[TradingRobotEnv.STATE_MARKET_BID]))/2
        # Стоимость корзины
        worth = num_usd + num_btc * course
        return worth

    # Вероятность исполнения сделки
    # Независит от Ask или Bid!
    # Bid = -1, Ask = 1
    def _execution_prob(self, our_type, market, our_order_level, our_order_size):
        k = self.COEFFICIENT_K
        prob = np.exp(-k * np.abs(market - our_order_level) / our_order_size)
        #         # Ask
        #         if (our_type == 1) & (our_order_level < market):
        #             prob = 1
        #         # Bid
        #         if (our_type == -1) & (our_order_level > market):
        #             prob = 1
        if our_type * (our_order_level - market) < 0:
            prob = 1
        return prob

    # _PNL = PnL
    def _PnL(self, condition_bid, condition_ask, level_bid, level_ask):
        our_bid_type = condition_bid  # 0 - не выставляем, 1 выставляем
        our_ask_type = condition_ask  # 0 - не выставляем, 1 выставляем
        our_bid_level = level_bid  # уровень bid
        our_ask_level = level_ask  # уровень ask
        order_size = self.ORDER_SIZE
        market_bid = float(self.state[TradingRobotEnv.STATE_MARKET_BID])
        market_ask = float(self.state[TradingRobotEnv.STATE_MARKET_ASK])
        mid = (market_ask + market_bid)/2

        # Ask type for execution_prob = -1
        # Bid type for execution_prob = 1
        pnl = 0.0
        if condition_bid == 1:
            pnl += (mid - our_bid_level) * self._execution_prob(self, -1, market_bid, our_bid_level, order_size)
        if condition_ask == 1:
            pnl += (our_ask_level - mid) * self._execution_prob(self, 1, market_ask, our_ask_level, order_size)

        return pnl

    # |Q|
    def _risk_function(self):
        num_btc = float(self.state[TradingRobotEnv.STATE_OUR_BTC])
        return num_btc

    # F
    def _objective_function(self, condition_bid, condition_ask, level_bid, level_ask):
        r = float(self.DERIVATIVE_OF_RISK)
        F = self._PnL(condition_bid, condition_ask, level_bid, level_ask) - r * self._risk_function()
        return F

    # a* = (Mk + s)/k
    # b* = (Mk - s)/k
    # Ищет максимум для PnL
    def func_max(self):
        mid = np.array(self.state[TradingRobotEnv.STATE_MARKET_ASK]) - np.array(
            self.state[TradingRobotEnv.STATE_MARKET_BID])
        k = np.array(self.COEFFICIENT_K)
        s = np.array(self.ORDER_SIZE)
        level_bid = mid - s / k
        level_ask = mid + s / k
        winning = 0.0
        condition_bid_ret = 0
        condition_ask_ret = 0
        level_bid_ret = 0.0
        level_ask_ret = 0.0
        for i in range(2):
            for j in range(2):
                condition_bid = i
                condition_ask = j
                pnl = self._PnL(condition_bid, condition_ask, level_bid, level_ask)
                if winning < pnl:
                    condition_bid_ret = condition_bid
                    condition_ask_ret = condition_ask
                    level_bid_ret = level_bid
                    level_ask_ret = level_ask

        return condition_bid_ret, condition_ask_ret, level_bid_ret, level_ask_ret

    # Оценивает действие
    def _eval_action(self, action):
        condition_bid = action[TradingRobotEnv.ACTION_BID]
        condition_ask = action[TradingRobotEnv.ACTION_ASK]
        level_bid = action[TradingRobotEnv.ACTION_BID_LEVEL]
        level_ask = action[TradingRobotEnv.ACTION_ASK_LEVEL]

        pnl = self._PnL(condition_bid, condition_ask, level_bid, level_ask)
        return pnl

    def step(self, action):
        self.last_action = action
        past_worth = self.state[TradingRobotEnv.STATE_NET_WORTH]

        our_usd, our_btc, done = Iterator.iterator()

        self.state_log.append(self.state + [net_worth,
                                            reward,
                                            portfolio[0],
                                            portfolio[1]])

            self.step_num += 1

        net_worth = self.calc_net_worth()

        # Calculate actions
        pnl = self.eval_action(action)

        if self.verbose:
            print(f"PnL: {pnl}")

        # Time to retire
        #         done = is_done

        # bid [0]-level, [1]-size
        if market_order[0] > 0:

        # ask [2]-level, [3]-size
        if

        # Calculate profit
        reward = net_worth - past_worth

        # Track progress
        if self.verbose:
            print(f"Networth: {net_worth}")
            print(f"*** End Step {self.step_num}: State={self.state}, \
          Reward={reward}")
        self.state_log.append(self.state + [reward])
        self.step_num += 1

        # Finish up
        step_state = [x for x in self.state]
        return step_state, reward, done, {}

    def reset(self):
        self.capital = TradingRobotEnv.STARTING_CAPITAL
        self.step_num = 0
        self.last_action = [0] * TradingRobotEnv.ACTION_ELEMENTS
        self.state = [0] * TradingRobotEnv.STATE_ELEMENTS
        self.state_log = []

        self.state[TradingRobotEnv.STATE_PORTFOLIO] = [1, 2]
        self.state[TradingRobotEnv.STATE_NET_WORTH] = []
        self.state[TradingRobotEnv.STATE_MARKET_BID] = []
        self.state[TradingRobotEnv.STATE_MARKET_ASK] = []
        self.state[TradingRobotEnv.STATE_MARKET_ORDER] = []
        self.state[TradingRobotEnv.STATE_PROFIT] = 0.0
        self.state[TradingRobotEnv.STATE_PROFIT] = 0.0

        #         return np.array(self.state)
        return self.state

    # возвращает изображение
    def render(self, mode='robot'):
        screen_width = 600
        screen_height = 400

        img = PIL.Image.new('RGB', (600, 400))
        d = PIL.ImageDraw.Draw(img)
        font = ImageFont.load_default()
        y = 0
        _, height = d.textsize("W", font)

        portfolio = self.state[TradingRobotEnv.STATE_PORTFOLIO]
        worth = self.state[TradingRobotEnv.STATE_NET_WORTH]

        net_worth = self.calc_net_worth()

        d.text((0, y), f"Porfolio: {portfolio}", fill=(0, 255, 0))
        y += height
        d.text((0, y), f"Net worth: {net_worth:,}", fill=(0, 255, 0))
        y += height * 2

        risk = risk_function()
        d.text((0, y), f"Portfolio risk measure: {risk}", fill=(0, 255, 0))

        return np.array(img)

    def close(self):
        pass
