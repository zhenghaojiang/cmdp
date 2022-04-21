from abc import ABC
from typing import Dict, Any, Union, Tuple

import gym
import numpy as np
from scipy import stats

from cmdp import cMDP, MDP
from utils.distributions import ConstantDistribution


class MDPCattle(MDP, ABC):
    def reset(self) -> np.ndarray:
        '''x_pp_init, x_p_init, x_init, m_init, h_init, p_init = self.config['initial_state']
        self.state = np.array(
            [x_pp_init, x_p_init, x_init, m_init, h_init, p_init]) # 6 states
        self.observation = np.concatenate((np.array([x_pp_init, x_p_init, x_init, m_init, h_init, p_init], dtype=np.float32),
                                           self.context), axis=0).flatten()'''
        x_pp_init, x_p_init, k_init, m_init, h_init, p_init = self.config['initial_state']
        t_init = 0
        self.state = np.array(
            [x_pp_init, x_p_init, k_init, m_init, h_init, p_init, t_init]) # 6 states
        self.observation = np.concatenate((np.array([x_pp_init, x_p_init, k_init, m_init, h_init, p_init, t_init], dtype=np.float32),
                                           self.context), axis=0).flatten()
        self.steps = 0
        self.done = False
        return self.observation

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        gamma_0, gamma_1, g, rho_h, rho_m, sigma_h, sigma_m, espilon_sq, mu_h, mu_m = self.context # 10 params
        # x_pp_curr, x_p_curr, x_curr, m_curr, h_curr, p_curr = self.state
        x_pp_curr, x_p_curr, k_curr, m_curr, h_curr, p_curr, t_curr = self.state
        if isinstance(action, np.ndarray):
            assert action.shape[0] == 1
            action = int(action[0])
        # discrete action denoting fraction of breeding stock
        # c_new = action * x_curr / (self.action_space.n - 1.)
        c_curr = action * k_curr * 0.7 / (self.action_space.n - 1.)
        # update x
        x_pp_new = x_p_curr
        # x_p_new = x_curr
        x_curr = k_curr - c_curr
        x_p_new = x_curr
        # x_new = x_curr + g * x_pp_curr - c_new
        k_new = k_curr + g * x_pp_curr - c_curr
        # update m
        m_new = (1 - rho_m) * mu_m + m_curr + np.random.normal(0, sigma_m, 1)[0]
        # update h
        h_new = (1 - rho_h) * mu_h + h_curr + np.random.normal(0, sigma_h, 1)[0]
        # h_new = (1 - rho_h) * mu_h + h_curr
        # update p
        p_new = 2.5 + p_curr + np.random.normal(0, 4, 1)[0]

        # reward
        '''reward = ((p_new - m_new) * c_new 
                    - h_new * x_new -gamma_0 * h_new * g * x_p_new - gamma_1 * h_new * g * x_pp_new 
                    - espilon_sq * (x_new ** 2 + x_p_new ** 2 + x_pp_new ** 2 + c_new ** 2))'''
        reward = ((p_curr - m_curr) * c_curr 
                    - h_curr * x_curr -gamma_0 * h_curr * g * x_p_curr - gamma_1 * h_curr * g * x_pp_curr 
                    - espilon_sq * (x_curr ** 2 + x_p_curr ** 2 + x_pp_curr ** 2 + c_curr ** 2))
        assert isinstance(reward, float)
        
        
        # add step
        self.steps += 1
        t_new = 0 * self.steps / self.config['max_steps']

        # new state and observations
        '''self.observation = np.concatenate((np.array([x_pp_new, x_p_new, x_new, m_new, h_new, p_new], dtype=np.float32),
                                           self.context), axis=0).flatten()'''
        self.observation = np.concatenate((np.array([x_pp_new, x_p_new, k_new, m_new, h_new, p_new, t_new], dtype=np.float32),
                                           self.context), axis=0).flatten()
        '''self.state = np.array([x_pp_new, x_p_new, x_new, m_new, h_new, p_new])'''
        self.state = np.array([x_pp_new, x_p_new, k_new, m_new, h_new, p_new, t_new])

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
                          ConstantDistribution(dim=10, constant_vector=np.array([1.0, 1.4, 0.95, 0.93, 0.7, .53, .40, 1E-8, 3.7, 6.3])))
        config['env_config'] = self.unpack_env_config(config)

        super().__init__(config=config)

    def unpack_env_config(self, config) -> Dict:
        env_config = config.get('env_config', {})
        env_config.setdefault('action_space', gym.spaces.Discrete(20))
        env_config.setdefault('observation_space', gym.spaces.Box(np.zeros((17,)),
                                                                   np.full((17,), np.inf),
                                                                   dtype=np.float32))
        env_config.setdefault('max_steps', 100)
        env_config.setdefault('initial_state', (100., 100., 200., 35., 5., 50.))

        return env_config

    def likelihood(self, obs_prev, action_prev, obs_curr, reward_prev):
        gamma_0, gamma_1, g, rho_h, rho_m, sigma_h, sigma_m, espilon_sq, mu_h, mu_m = self.context
        x_pp_prev, x_p_prev, k_prev, m_prev, _, p_prev = obs_prev[:6]
        _, _, _, m_curr, h_curr, p_curr = obs_curr[:6]
        r_prev = reward_prev
        c_prev = action_prev * k_prev * 0.8 / (self.action_space.n - 1.)
        x_prev = k_prev - c_prev
        h_prev = (((p_prev - m_prev) * c_prev - reward_prev
                    - espilon_sq * (x_prev ** 2 + x_p_prev ** 2 + x_pp_prev ** 2 + c_prev ** 2))
                    / (x_prev + gamma_0 * g * x_p_prev + gamma_1 * g * x_pp_prev))
        
        eps_p = p_curr - 2.5 - p_prev
        eps_m = m_curr - (1 - rho_m) * mu_m - m_prev
        eps_h = h_curr - (1 - rho_h) * mu_h - h_prev

        likelihood_p = stats.norm.pdf(eps_p, loc=0, scale=4)
        likelihood_m = stats.norm.pdf(eps_m, loc=0, scale=sigma_m)
        likelihood_h = stats.norm.pdf(eps_h, loc=0, scale=sigma_h)
        likelihood = likelihood_p * likelihood_m * likelihood_h

        return np.nan_to_num(likelihood, copy=False)

