import numpy as np
import ray
from ray.rllib.agents import ppo

from mdps.whited import cMDPWhited
from utils.distributions import ConstantDistribution, ParticleDistribution, UniformDistribution

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

# Filter function
def filter_context(solver_,
                   context_distribution_,
                   gt_obs_arr_,
                   T_,
                   N_
                   ):
    state_arr_ = np.ones((N_,2))
    action_arr_ = np.zeros((N_,))
    context_history_ = []
    for t_ in range(T_):
        # we only use the first 5 steps of the cartpole steps to reduce effect of different episode lengths
        qs_ = np.zeros((N_,))
        for n_ in range(N_):
            context_ = context_distribution_.particles[n_]
            c_local_ = {"context_distribution":
                           ConstantDistribution(dim=5,
                                                constant_vector=context_)
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
            q = env_.likelihood(gt_obs_arr_[t_], action_arr_[n_], obs_, reward_)
            qs_[n_] = q
            state_arr_[n_] = np.copy(env_.mdp.state)
        if t_ >= 1:
            # truncated importance sampling; [https://arxiv.org/pdf/1905.09800.pdf]
            qs_ = np.clip(qs_, 0, np.percentile(qs_, 90))
            if qs_.sum() == 0:
                continue
            qs_ = qs_ / qs_.sum()
            resample_index_ = context_distribution_.resample_particles_from_probability(p=qs_)
            p_temp_ = context_distribution_.particles
            p_noise_ = np.random.normal(loc=0, scale=p_temp_.std(axis=0), size=p_temp_.shape) * 0.05
            context_distribution_.particles += p_noise_
            context_distribution_.particles = np.clip(context_distribution_.particles, 0.0, 1.0)
            state_arr_ = state_arr_[resample_index_]
            action_arr_ = action_arr_[resample_index_]
        if t_ % 25 == 0:
            print("round", t_, "posterior mean", context_distribution_.particles[:, 2].mean())
        context_history_ += [context_distribution_.particles.copy()]
    return context_history_, context_distribution_

# 

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
        ConstantDistribution(dim=5, constant_vector=np.array([0.98, 0.15, 0.5, 0.7, 0.15]))
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
                                 lower_bound_vector=np.array([0.98, 0.15, 0.4, 0.7, 0.15]),
                                 upper_bound_vector=np.array([0.98, 0.15, 0.8, 0.7, 0.15]))}

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
N = 500
T = 100

gamma = np.ones((N,)) * 0.98
delta = np.ones((N,)) * 0.15
theta = np.random.uniform(0.4, 0.8, size=(N,))
rho = np.ones((N,)) * 0.7
sigma = np.ones((N,)) * 0.15

context_particles = np.abs(np.vstack((gamma, delta, theta, rho, sigma)).T)
context_distribution = ParticleDistribution(dim=5, particles=context_particles, n_particles=N)

n_trajectory = 15
n_round = 3
which_param = 2

# simulation
context_exact = None
context_mean_exact = None
context_mis = None
context_mean_mis = None
context_uniform = None
context_mean_uniform = None
context_imp = None
context_mean_imp = None

for i_ in range(n_trajectory):

    gt_obs_arr, _, _ = get_rollouts(expert, config=c)

    for j_ in range(n_round):

        # exact
        context_distribution = ParticleDistribution(
            dim=5, particles=context_particles, n_particles=N)
        context_history, _ = filter_context(expert,
                                                context_distribution,
                                                gt_obs_arr,
                                                T,
                                                N
                                                )
        context_curr = context_history[-1][:, which_param]
        if context_exact is None:
            context_exact = context_curr
            context_mean_exact = [context_curr.mean()]
        else:
            context_exact = np.concatenate((context_exact, context_curr))
            context_mean_exact += [context_curr.mean()]
        print("exact", "round", (i_ + 1, j_ + 1))

        # mis
        context_distribution = ParticleDistribution(
            dim=5, particles=context_particles, n_particles=N)
        context_history, _ = filter_context(solver_mis,
                                                context_distribution,
                                                gt_obs_arr,
                                                T,
                                                N
                                                )
        context_curr = context_history[-1][:, which_param]
        if context_mis is None:
            context_mis = context_curr
            context_mean_mis = [context_curr.mean()]
        else:
            context_mis = np.concatenate((context_mis, context_curr))
            context_mean_mis += [context_curr.mean()]
        print("mis", "round", (i_ + 1, j_ + 1))

        # uniform
        context_distribution = ParticleDistribution(
            dim=5, particles=context_particles, n_particles=N)
        context_history, _ = filter_context(solver_uniform,
                                                context_distribution,
                                                gt_obs_arr,
                                                T,
                                                N
                                                )
        context_curr = context_history[-1][:, which_param]
        if context_uniform is None:
            context_uniform = context_curr
            context_mean_uniform = [context_curr.mean()]
        else:
            context_uniform = np.concatenate((context_uniform, context_curr))
            context_mean_uniform += [context_curr.mean()]
        print("uniform", "round", (i_ + 1, j_ + 1))

        # imp
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

            context_history_imp_, imp_context_distribution = filter_context(imp_solver,
                                                                            imp_context_distribution,
                                                                            gt_obs_arr,
                                                                            T,
                                                                            N
                                                                            )

            imp_solver.workers.foreach_worker(
                        lambda ev: ev.foreach_env(
                            lambda env: env.set_task(imp_context_distribution)))
            if update_round < 3:
                for _ in range(5):
                    imp_solver.train()
        context_curr = context_history_imp_[-1][:,which_param]
        if context_imp is None:
            context_imp = context_curr
            context_mean_imp = [context_curr.mean()]
        else:
            context_imp = np.concatenate((context_imp, context_curr))
            context_mean_imp += [context_curr.mean()]
        print("imp", "round", (i_ + 1, j_ + 1))

context_mean_exact = np.array(context_mean_exact)
context_mean_mis = np.array(context_mean_mis)
context_mean_uniform = np.array(context_mean_uniform)
context_mean_imp = np.array(context_mean_imp)

np.save("context_exact.npy", context_exact)
np.save("context_mean_exact.npy", context_mean_exact)
np.save("context_mis.npy", context_mis)
np.save("context_mean_mis.npy", context_mean_mis)
np.save("context_uniform.npy", context_uniform)
np.save("context_mean_uniform.npy", context_mean_uniform)
np.save("context_imp.npy", context_imp)
np.save("context_mean_imp.npy", context_mean_imp)



