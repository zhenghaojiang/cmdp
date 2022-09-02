from abc import ABC
from typing import Dict, Any, Union

import gym
import numpy as np
from scipy.special import expit, logit
from math import exp, log

from cmdp import cMDP, MDP
from utils.distributions import ConstantDistribution
from utils.math_fn import log_ar1


class MDPWhited(MDP, ABC):
    def reset(self) -> np.ndarray:
        k_init, transz_init = self.config['initial_state']
        self.state = np.array([k_init, transz_init])
        self.observation = np.concatenate((np.array([k_init, transz_init], dtype=np.float32),
                                           self.context), axis=0).flatten()
        self.steps = 0
        self.done = False
        return self.observation

    # action is investment I_t over capital k_
    def step(self, action: Union[float, np.ndarray]):
        # Unpack context
        gamma, delta, theta, rho, sigma = self.context
        k_curr, transz_curr = self.state
        z_curr = 1 / (1 + exp(-transz_curr)) + 0.5
        if isinstance(action, np.ndarray):
            assert action.shape[0] == 1
            # Discrete action denoting fraction of capital to invest
            action = int(action[0])
        i_curr = action * k_curr / (self.action_space.n - 1.)
        # update z
        z_new = self.config['shock_process_fn'](z_curr, rho, sigma)
        # update k
        k_new = (1 - delta) * k_curr + i_curr
        # reward function
        reward = self.config['pi_fn'](
            k_curr, z_curr, theta) - self.config['psi_fn'](i_curr, k_curr) - i_curr
        assert isinstance(reward, float)
        # new state and observations
        log((z_new - 0.5)/(1.5 - z_new))
        self.observation = np.concatenate((np.array([k_new, log((z_new - 0.5)/(1.5 - z_new))], dtype=np.float32), self.context),
                                          axis=0).flatten()
        self.state = np.array([k_new, log((z_new - 0.5)/(1.5 - z_new))])
        # add step
        self.steps += 1
        # decide to end or not
        if self.steps == self.config['max_steps']:
            self.done = True
        return self.observation, reward, self.done, {}


class cMDPWhited(cMDP, ABC):
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        config.setdefault('mdp_type', MDPWhited)
        config.setdefault('context_distribution',
                          ConstantDistribution(dim=5, constant_vector=np.array([0.98, 0.15, 0.7, 0.7, 0.15])))
        config['env_config'] = self.unpack_env_config(config)

        super().__init__(config=config)

    def unpack_env_config(self, config):
        env_config = config.get('env_config', {})
        env_config.setdefault('action_space', gym.spaces.Discrete(20))
        env_config.setdefault('observation_space', gym.spaces.Box(np.zeros((7,)),
                                                                  np.full(
                                                                      (7,), np.inf),
                                                                  dtype=np.float32))
        env_config.setdefault('max_steps', 100)
        # (k_0, z_0) time 0 capital and shock
        env_config.setdefault('initial_state', (1., 0.))
        # investment adjustment cost function psi(k, z)
        env_config.setdefault('psi_fn', lambda i, k: 0.)
        env_config.setdefault('pi_fn', lambda k, z, theta: z * (
            k ** theta))  # profit function pi(k, z) = z*k**theta
        # z process: z(z0, rho, sigma)
        env_config.setdefault('shock_process_fn', log_ar1)
        return env_config
