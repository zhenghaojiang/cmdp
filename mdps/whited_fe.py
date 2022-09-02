from abc import ABC
from cmath import exp
from typing import Dict, Any, Union

import gym
import numpy as np
from scipy import stats
from scipy.special import expit
from math import log

from cmdp import cMDP, MDP
from utils.distributions import ConstantDistribution
from utils.math_fn import log_ar1


class MDPWhited(MDP, ABC):
    def reset(self) -> np.ndarray:
        k_init, z_init, add_init = self.config['initial_state']
        self.state = np.array([k_init, z_init, add_init])
        self.observation = np.concatenate((np.array([k_init, z_init, add_init], dtype=np.float32),
                                           self.context), axis=0).flatten()
        self.steps = 0
        self.done = False
        return self.observation

    def step(self, action: Union[float, np.ndarray]):  # action is investment I_t over capital k_
        # Unpack context
        gamma, delta, theta, rho, sigma = self.context
        k_curr, z_curr, add_curr = self.state
        if isinstance(action, np.ndarray):
            assert action.shape[0] == 1
            action = int(action[0])  # Discrete action denoting fraction of capital to invest
        i_curr = action * k_curr / (self.action_space.n - 1.)
        # update z
        z_new = self.config['shock_process_fn'](z_curr, rho, sigma)
        # update k
        k_new = (1 - delta) * k_curr + i_curr
        # reward function
        reward = self.config['pi_fn'](k_curr, z_curr, theta) - self.config['psi_fn'](i_curr, k_curr) - i_curr
        assert isinstance(reward, float)
        # new state and observations
        self.observation = np.concatenate((np.array([k_new, z_new, expit(z_new)], dtype=np.float32), self.context),
                                          axis=0).flatten()
        self.state = np.array([k_new, z_new, expit(z_new)])
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
        env_config.setdefault('observation_space', gym.spaces.Box(-np.ones((8,)),
                                                                   np.full((8,), np.inf),
                                                                   dtype=np.float32))
        env_config.setdefault('max_steps', 100)
        env_config.setdefault('initial_state', (1., 1., expit(1.)))  # (k_0, z_0) time 0 capital and shock
        env_config.setdefault('psi_fn', lambda i, k: 0.)  # investment adjustment cost function psi(k, z)
        env_config.setdefault('pi_fn', lambda k, z, theta: z * (
                k ** theta))  # profit function pi(k, z) = z*k**theta
        env_config.setdefault('shock_process_fn', log_ar1)  # z process: z(z0, rho, sigma)
        return env_config
    
    def likelihood(self, obs_prev, action_prev, obs_curr, reward_prev):
        gamma, delta, theta, rho, sigma = self.context
        k_prev, _ = obs_prev[:2]
        _, z_curr = obs_curr[:2]
        r_prev = reward_prev
        I_prev = action_prev * k_prev / (self.action_space.n - 1.)
        z_prev = (r_prev+I_prev+self.env_config['psi_fn'](I_prev,k_prev))/(k_prev**theta)
        
        eps = np.log(z_curr)-rho*np.log(z_prev)
        likelihood = stats.norm.pdf(eps, loc=0, scale=sigma)
        # print("z_prev:", z_prev, "z_curr:", z_curr, "eps:", eps, "likelihood:", likelihood, "rho", rho)
        return np.nan_to_num(likelihood)
