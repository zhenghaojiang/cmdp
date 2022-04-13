from abc import ABC
from typing import Dict, Any, Union

import gym
import numpy as np
from scipy import stats

from cmdp import cMDP, MDP
from utils.distributions import ConstantDistribution
from utils.math_fn import log_ar1


class MDPWhited_cash(MDP, ABC):
    def reset(self) -> np.ndarray:
        k_init, z_init, p_init = self.config['initial_state']
        self.state = np.array([k_init, z_init, p_init])
        self.observation = np.concatenate((np.array([k_init, z_init, p_init], dtype=np.float32),
                                           self.context), axis=0).flatten()
        self.steps = 0
        self.done = False
        return self.observation

    # action is investment I_t over capital k_
    def step(self, action: Union[float, np.ndarray]):
        # Unpack context
        gamma, delta, theta, rho, sigma, eta_0, eta_1 = self.context
        k_curr, z_curr, p_curr = self.state
        # investment
        i_curr = action[0] * k_curr / (self.action_space.nvec[0] - 1.)
        j_curr = action[1] * p_curr / (self.action_space.nvec[1] - 1.)
        # update z
        z_new = self.config['shock_process_fn'](z_curr, rho, sigma)
        # update k
        k_new = (1 - delta) * k_curr + i_curr
        # update p
        p_new = (1 + (1 / gamma - 1) * 0.8) * (p_curr + j_curr)
        # cash flow
        cash = 0.8 * self.config['pi_fn'](k_curr, z_curr, theta) + 0.2 * delta * k_curr - \
            self.config['psi_fn'](i_curr, k_curr) - i_curr - j_curr
        assert isinstance(cash, float)
        # cost of external finance
        cost_ex = 0
        if cash < 0:
            cost_ex = eta_0 + eta_1 * cash
        # reward function
        reward = cash + cost_ex
        # new state and observations
        self.observation = np.concatenate((np.array([k_new, z_new, p_new], dtype=np.float32), self.context),
                                          axis=0).flatten()
        self.state = np.array([k_new, z_new, p_new])
        # add step
        self.steps += 1
        # decide to end or not
        if self.steps == self.config['max_steps']:
            self.done = True
        return self.observation, reward, self.done, {}


class cMDPWhited_cash(cMDP, ABC):

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        config.setdefault('mdp_type', MDPWhited_cash)
        config.setdefault('context_distribution',
                          ConstantDistribution(dim=7, constant_vector=np.array([0.98, 0.15, 0.7, 0.7, 0.15, 0.0, 0.07])))
        config['env_config'] = self.unpack_env_config(config)

        super().__init__(config=config)

    def unpack_env_config(self, config):
        env_config = config.get('env_config', {})
        env_config.setdefault('action_space', gym.spaces.MultiDiscrete([20, 20]))
        env_config.setdefault('observation_space', gym.spaces.Box(np.zeros((10,)),
                                                                  np.full(
                                                                      (10,), np.inf),
                                                                  dtype=np.float32))
        env_config.setdefault('max_steps', 100)
        # (k_0, z_0) time 0 capital and shock
        env_config.setdefault('initial_state', (1., 1., 1.))
        # investment adjustment cost function psi(k, z)
        env_config.setdefault('psi_fn', lambda i, k: 0.)
        env_config.setdefault('pi_fn', lambda k, z, theta: z * (
            k ** theta))  # profit function pi(k, z) = z*k**theta
        # z process: z(z0, rho, sigma)
        env_config.setdefault('shock_process_fn', log_ar1)
        return env_config

    def likelihood(self, obs_prev, action_prev, obs_curr, reward_prev):
        pass