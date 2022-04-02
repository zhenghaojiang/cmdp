from abc import ABC
from typing import Dict, Any, Union, Tuple

import gym
import numpy as np
from scipy import stats

from cmdp import cMDP, MDP
from utils.distributions import ConstantDistribution


class MDPCattle(MDP, ABC):
    def reset(self) -> np.ndarray:
        x_pp_init, x_p_init, x_init, m_init, h_init, p_init = self.config['initial_state']
        self.state = np.array(
            [x_pp_init, x_p_init, x_init, m_init, h_init, p_init]) # 6 states
        self.observation = np.concatenate((np.array([x_pp_init, x_p_init, x_init, m_init, h_init, p_init], dtype=np.float32),
                                           self.context), axis=0).flatten()
        self.steps = 0
        self.done = False
        return self.observation

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        gamma_0, gamma_1, g, rho_h, rho_m, sigma_h, sigma_m, espilon_sq, mu_h, mu_m = self.context # 10 params
        x_pp_curr, x_p_curr, x_curr, m_curr, h_curr, p_curr = self.state
        if isinstance(action, np.ndarray):
            assert action.shape[0] == 1
            action = int(action[0])
        # discrete action denoting fraction of breeding stock
        c_curr = action * x_curr / (self.action_space.n - 1.)
        # update x
        x_pp_new = x_p_curr
        x_p_new = x_curr
        x_new = x_curr + g * x_pp_curr - c_curr
        # update m
        m_new = (1 - rho_m) * mu_m + m_curr + np.random.normal(0, sigma_m, 1)[0]
        # update h
        h_new = (1 - rho_h) * mu_h + h_curr + np.random.normal(0, sigma_h, 1)[0]
        # update p
        p_new = p_curr + np.random.normal(0, 1, 1)[0]
        # reward
        reward = ((p_curr - m_curr) * c_curr 
                    - h_curr * x_curr -gamma_0 * h_curr * g * x_p_curr - gamma_1 * h_curr * g * x_pp_curr 
                    - espilon_sq * (x_curr ** 2 + x_p_curr ** 2 + x_pp_curr ** 2 + c_curr ** 2))
        assert isinstance(reward, float)
        # new state and observations
        self.observation = np.concatenate((np.array([x_pp_new, x_p_new, x_new, m_new, h_new, p_new], dtype=np.float32),
                                           self.context), axis=0).flatten()
        self.state = np.array([x_pp_new, x_p_new, x_new, m_new, h_new, p_new])
        # add step
        self.steps += 1
        # decide to end or not
        if self.steps == self.config['max_steps']:
            self.done = True                     
        return self.observation, reward, self.done, {}


class cMDPCattle(cMDP, ABC):
    def __init__(self, config: Dict = None):
        if config is None:
            config = {}
        config.setdefault('mdp_type', MDPCattle)
        config.setdefault('context_distribution',
                          ConstantDistribution(dim=10, constant_vector=np.array([1.0, 1.4, 0.95, 0.95, 0.7, 0.3, 0.5, 1E-8, 37, 63])))
        config['env_config'] = self.unpack_env_config(config)

        super().__init__(config=config)

    def unpack_env_config(self, config) -> Dict:
        env_config = config.get('env_config', {})
        env_config.setdefault('action_space', gym.spaces.Discrete(100))
        env_config.setdefault('observation_space', gym.spaces.Box(np.zeros((16,)),
                                                                   np.full((16,), np.inf),
                                                                   dtype=np.float32))
        env_config.setdefault('max_steps', 100)
        env_config.setdefault('initial_state', (100., 100., 1000., 1., 1., 50.))

        return env_config

    def likelihood(self):
        pass
