import numpy as np
import matplotlib.pyplot as plt
import sys
import ray
from ray.rllib.agents import ppo

from mdps.whited import cMDPWhited
from utils.distributions import ConstantDistribution, ParticleDistribution, UniformDistribution
from estimators.whited_estimator import EstWhited

def get_rollouts(solver_, config):
    """Generate rollouts from a given solver and MDP(c)"""
    env_ = cMDPWhited(config=config)
    done_ = False
    obs_ = env_.reset()
    # run until episode ends
    gt_obs_arr_ = obs_
    gt_act_arr_ = None
    gt_rew_arr_ = None
    while not done_:
        action_ = solver_.compute_single_action(obs_)
        obs_, rewawrd_, done_, _ = env_.step(action_)
        gt_obs_arr_ = np.vstack((gt_obs_arr_, obs_))
        if gt_act_arr_ is None:
            gt_act_arr_ = [action_]
            gt_rew_arr_ = [rewawrd_]
        else:
            gt_act_arr_ += [action_]
            gt_rew_arr_ += [rewawrd_]

    gt_act_arr_ = np.array(gt_act_arr_)
    gt_rew_arr_ = np.array(gt_rew_arr_)
    return gt_obs_arr_[1:,:], gt_act_arr_, gt_rew_arr_

# exact solver
c = {"context_distribution":
        ConstantDistribution(dim=5, constant_vector=np.array([0.98, 0.15, 0.7, 0.7, 0.15]))
    }

ray.shutdown()
ray.init()

expert = ppo.PPOTrainer(env=cMDPWhited, config={
    "env_config": c,
    "framework": "torch",  # config to pass to env class
})

rews = []
for eps in range(35):
    res = expert.train()
    if eps % 5 == 0:
        print(eps, res['episode_reward_mean'])
    rews += [res['episode_reward_mean']]

# misspecified solver
c_mis = {"context_distribution":
        ConstantDistribution(dim=5, constant_vector=np.array([0.98, 0.15, 0.7, 0.7, 0.3]))
    }

ray.shutdown()
ray.init()

solver_mis = ppo.PPOTrainer(env=cMDPWhited, config={
    "env_config": c_mis,
    "framework": "torch",  # config to pass to env class
})

rews = []
for eps in range(35):
    res = solver_mis.train()
    if eps % 5 == 0:
        print(eps, res['episode_reward_mean'])
    rews += [res['episode_reward_mean']]

# uniform solver
c_uniform = {'context_distribution':
             UniformDistribution(dim=5,
                                 lower_bound_vector=np.array([0.98, 0.15, 0.7, 0.7, 0.05]),
                                 upper_bound_vector=np.array([0.98, 0.15, 0.7, 0.7, 0.5]))}

ray.shutdown()
ray.init()
solver_uniform = ppo.PPOTrainer(env=cMDPWhited, config={
                                                    "env_config": c_uniform,
                                                    "framework": "torch",  # config to pass to env class
                                                })

rews = []
for eps in range(35):
    res = solver_uniform.train()
    if eps % 5 == 0:
        print(eps, res['episode_reward_mean'])
    rews += [res['episode_reward_mean']]

# Pre-determined params
gt_obs_arr, _, _ = get_rollouts(expert, config=c)

N = 1000

gamma = np.ones((N,)) * 0.98
delta = np.ones((N,)) * 0.15
theta = np.ones((N,)) * 0.7
rho = np.ones((N,)) * 0.7
sigma = np.random.uniform(0.05, 0.5, size=(N,))

# exact est

context_particles = np.abs(np.vstack((gamma, delta, theta, rho, sigma)).T)
context_distribution = ParticleDistribution(dim=5, particles=context_particles, n_particles=N)

est_config = {
    'solver': expert, 
    'context_distribution': context_distribution, 
    'observation_array': gt_obs_arr,
    'estimate_index': 4, 
    'particle_number': N,
    'noise_rate': 0.25,  
}

estimator = EstWhited(config=est_config)
exact, exact_mean = estimator.estiamte()

# mis est

context_particles = np.abs(np.vstack((gamma, delta, theta, rho, sigma)).T)
context_distribution = ParticleDistribution(dim=5, particles=context_particles, n_particles=N)

est_config = {
    'solver': solver_mis, 
    'context_distribution': context_distribution, 
    'observation_array': gt_obs_arr,
    'estimate_index': 4, 
    'particle_number': N, 
    'noise_rate': 0.25,  
}

estimator = EstWhited(config=est_config)
mis, mis_mean = estimator.estiamte()

# uniform est

context_particles = np.abs(np.vstack((gamma, delta, theta, rho, sigma)).T)
context_distribution = ParticleDistribution(dim=5, particles=context_particles, n_particles=N)

est_config = {
    'solver': solver_uniform, 
    'context_distribution': context_distribution, 
    'observation_array': gt_obs_arr,
    'estimate_index': 4, 
    'particle_number': N, 
    'noise_rate': 0.25,  
}

estimator = EstWhited(config=est_config)
uniform, uniform_mean = estimator.estiamte()

# imp est
context_particles = np.abs(np.vstack((gamma, delta, theta, rho, sigma)).T)
imp_context_distribution = ParticleDistribution(dim=5, particles=context_particles, n_particles=N)

ray.shutdown()
ray.init()

imp_solver = ppo.PPOTrainer(env=cMDPWhited, config={
                                                "env_config":  {"context_distribution": imp_context_distribution},
                                                "framework": "torch",
                                            })

for update_round in range(4):
    # burn in training
    if update_round == 0:
        for i in range(20):
            imp_solver.train()
    
    est_config = {
        'solver': imp_solver, 
        'context_distribution': imp_context_distribution, 
        'observation_array': gt_obs_arr,
        'estimate_index': 4, 
        'particle_number': N, 
        'noise_rate': 0.25,  
    }

    estimator = EstWhited(config=est_config)
    context_history_imp, imp_context_distribution = estimator.filter_context()

    imp_solver.workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: env.set_task(imp_context_distribution)))
    if update_round < 3:
        for _ in range(5):
            imp_solver.train()
imp = context_history_imp[-1][:,est_config['estimate_index']]
imp_mean = [imp.mean()]

# end
print("end")
print([exact_mean, mis_mean, uniform_mean, imp_mean])

fig, ax = plt.subplots()
HIST_BINS = np.linspace(0.0, 0.5, 100)
ax.hist(exact, HIST_BINS, lw=1,
        ec="blue", fc="blue", alpha=0.5)
ax.hist(mis, HIST_BINS, lw=1,
        ec="green", fc="green", alpha=0.5)
ax.hist(uniform, HIST_BINS, lw=1,
        ec="gold", fc="gold", alpha=0.5)
ax.hist(imp, HIST_BINS, lw=1,
        ec="red", fc="red", alpha=0.5)
plt.axvline(x=0.15, alpha=0.3, color='black', linestyle='--')
plt.legend(['ground truth: 0.15', 'exact', 'misspecified', 'uniform', 'importance sampling'])
fig.set_size_inches(12, 8)
plt.title('posterior distribution trained in differently sampled context (Basic)')
plt.savefig('basic_hist_sigma.pdf') 





