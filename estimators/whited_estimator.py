import numpy as np
from typing import Type, Dict, Any, Tuple, Union
# import ray
# from ray.rllib.agents import ppo
# import logging

from mdps.whited import cMDPWhited
from utils.distributions import ConstantDistribution, ParticleDistribution, UniformDistribution


class EstWhited():
    def __init__(self, config: Dict = None):
        if config is None:
            config = {}
        assert all(k in config for k in ['solver', 'context_distribution', 'observation_array'])

        self.solver = config['solver']
        self.context_distribution = config['context_distribution']
        self.observation_array = config['observation_array']

        config.setdefault('particle_number', 1000)
        config.setdefault('max_steps', 100)
        config.setdefault('estimate_index', 2) # estimate theta by default
        config.setdefault('clip_quantile', 90)
        config.setdefault('noise_rate', 0.1)
        self.N = config['particle_number']
        self.T = config['max_steps']
        self.estimate_index = config['estimate_index']
        self.clip_quantile = config['clip_quantile']
        self.noise_rate = config['noise_rate']

    def filter_context(self) -> Tuple[list, Any]:
        N_ = self.N
        T_ = self.T
        solver_ = self.solver
        context_distribution_ = self.context_distribution
        estimate_index_ = self.estimate_index
        gt_obs_arr_ = self.observation_array

        state_arr_ = np.ones((N_,2))
        action_arr_ = np.zeros((N_,))
        context_history_ = []
        for t_ in range(T_):
            # we only use the first 5 steps of the cartpole steps to reduce effect of different episode lengths
            qs_ = np.zeros((N_,))
            for n_ in range(N_):
                context_ = context_distribution_.particles[n_]
                c_local_ = {"context_distribution":
                            ConstantDistribution(dim=5, constant_vector=context_)
                            }
                env_ = cMDPWhited(config=c_local_)
                obs_ = env_.reset()
                if t_ > 0:
                    env_.mdp.state = state_arr_[n_]
                    obs_ = np.concatenate((np.array(env_.mdp.state), context_), axis=0).flatten()
                action_ = solver_.compute_single_action(obs_)
                obs_, reward_, _, _ = env_.step(action_)
                # estimate likelihood if r >= 1
                action_arr_[n_] = action_
                q = env_.likelihood(gt_obs_arr_[t_], action_arr_[n_], obs_, reward_, estimate_index_)
                qs_[n_] = q
                state_arr_[n_] = np.copy(env_.mdp.state)
            if t_ >= 1:
                # truncated importance sampling; [https://arxiv.org/pdf/1905.09800.pdf]
                qs_ = np.clip(qs_, 0, np.percentile(qs_, self.clip_quantile))
                if qs_.sum() == 0:
                    continue
                qs_ = qs_ / qs_.sum()
                resample_index_ = context_distribution_.resample_particles_from_probability(p=qs_)
                p_temp_ = context_distribution_.particles
                p_noise_ = np.random.normal(loc=0, scale=p_temp_.std(axis=0), size=p_temp_.shape) * self.noise_rate
                context_distribution_.particles += p_noise_
                context_distribution_.particles = np.clip(context_distribution_.particles, 0.0, 1.0)
                state_arr_ = state_arr_[resample_index_]
                action_arr_ = action_arr_[resample_index_]
            if t_ % 25 == 0:
                print("round", t_, "posterior mean", context_distribution_.particles[:, estimate_index_].mean())
            context_history_ += [context_distribution_.particles.copy()]
        return context_history_, context_distribution_

    def estiamte(self):
        context_history_, _ = self.filter_context()
        context_ = context_history_[-1][:, self.estimate_index]
        context_mean_ = [context_.mean()]
        return context_, context_mean_