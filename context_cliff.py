import numpy as np
import ray
from ray.rllib.agents import ppo

from mdps.cliff import ContextualCliff
from utils.distributions import ConstantDistribution, ParticleDistribution, UniformDistribution

def get_rollouts(solver_, config):
    """Generate rollouts from a given solver and MDP(c)"""
    env_ = ContextualCliff(config=config)
    done_ = False
    obs_ = env_.reset()
    # run until episode ends
    gt_obs_arr_ = None
    gt_act_arr_ = None
    while not done_:
        action_ = solver_.compute_single_action(obs_)
        obs_, _, done_, _ = env_.step(action_)
        if gt_obs_arr_ is None:
            gt_obs_arr_ = obs_
            gt_act_arr_ = [action_]
        else:
            gt_obs_arr_ = np.vstack((gt_obs_arr_, obs_))
            gt_act_arr_ += [action_]

    gt_act_arr_ = np.array(gt_act_arr_)
    return gt_obs_arr_, gt_act_arr_

def filter_context(solver_,
                   context_distribution_,
                   gt_obs_arr_,
                   T_,
                   N_
                   ):
    state_arr_ = np.zeros((N_,))
    action_arr_ = np.zeros((N_,))
    context_history_ = []
    for t_ in range(T_):
        # we only use the first 5 steps of the cartpole steps to reduce effect of different episode lengths
        qs_ = np.zeros((N_,))
        for n_ in range(N_):
            context_ = context_distribution_.particles[n_]
            c_local_ = {'context_distribution':
                           ConstantDistribution(dim=6,
                                                constant_vector=context_)}
            env_ = ContextualCliff(config=c_local_)
            obs_ = env_.reset()
            if t_ > 0:
                env_.mdp.x = state_arr_[n_]
                obs_ = np.concatenate((np.array([env_.mdp.x]), context_), axis=0).flatten()
            action_ = solver_.compute_single_action(obs_)
            obs_, _, done_, _ = env_.step(action_)
            # estimate likelihood if r >= 1
            if t_ >= 1:
                q = env_.likelihood(gt_obs_arr_[t_ - 1], action_arr_[n_], obs_)
                qs_[n_] = q
            state_arr_[n_] = np.copy(env_.mdp.x)
            action_arr_[n_] = action_
        if t_ >= 1:
            # truncated importance sampling; [https://arxiv.org/pdf/1905.09800.pdf]
            qs_ = np.clip(qs_, 0, np.percentile(qs_, 90))
            qs_ = qs_ / qs_.sum()
            resample_index_ = context_distribution_.resample_particles_from_probability(p=qs_)
            p_temp_ = context_distribution_.particles
            p_noise_ = np.random.normal(loc=0, scale=p_temp_.std(axis=0), size=p_temp_.shape) * 0.05
            context_distribution_.particles += p_noise_
            state_arr_ = state_arr_[resample_index_]
            action_arr_ = action_arr_[resample_index_]
        if t_ % 25 == 0:
            print("round", t_, "posterior mean", context_distribution_.particles[:, 1].mean())
        context_history_ += [context_distribution_.particles.copy()]
    return context_history_, context_distribution_

def multi_round_context(sim_solver_,
                        sim_config_,
                        solver_,
                        context_particles_,
                        T_,
                        N_,
                        n_trajectory_,
                        n_round_,
                        which_param_
                        ):
    context_ = None
    context_mean_ = None
    for i_ in range(n_trajectory_):
        for _ in range(10):
            gt_obs_arr_, _ = get_rollouts(sim_solver_, config=sim_config_)
            if gt_obs_arr_.shape[0] == 100:
                break
        assert gt_obs_arr_.shape[0] == 100
        for j_ in range(n_round_):
            context_distribution_ = ParticleDistribution(
                dim=6, particles=context_particles_, n_particles=N_)
            context_history_, _ = filter_context(solver_,
                                                 context_distribution_,
                                                 gt_obs_arr_,
                                                 T_,
                                                 N_
                                                 )
            context_curr_ = context_history_[-1][:, which_param_]
            if context_ is None:
                context_ = context_curr_
                context_mean_ = [context_curr_.mean()]
            else:
                context_ = np.concatenate((context_, context_curr_))
                context_mean_ += [context_curr_.mean()]
            print("round", (i_ + 1, j_ + 1))
    context_mean_ = np.array(context_mean_)

    return context_, context_mean_

def multi_round_imp_context(sim_solver_,
                            sim_config_,
                            context_particles_,
                            T_,
                            N_,
                            n_trajectory_,
                            n_round_,
                            which_param_
                            ):
    context_ = None
    context_mean_ = None
    for i_ in range(n_trajectory_):
        
        for _ in range(10):
            gt_obs_arr_, _ = get_rollouts(sim_solver_, config=sim_config_)
            if gt_obs_arr_.shape[0] == 100:
                break
        
        assert gt_obs_arr_.shape[0] == 100

        for j_ in range(n_round_):
            imp_context_distribution_ = ParticleDistribution(dim=6, particles=context_particles_, n_particles=N_)

            ray.shutdown()
            ray.init()

            imp_solver_ = ppo.PPOTrainer(env=ContextualCliff, config={
                                                            "env_config":  {"context_distribution": imp_context_distribution_},
                                                            "framework": "torch",
                                                        })

            for update_round in range(3):
                # burn in training
                if update_round == 0:
                    for i in range(15):
                        imp_solver_.train()

                context_history_imp_, imp_context_distribution_ = filter_context(imp_solver_,
                                                                                imp_context_distribution_,
                                                                                gt_obs_arr_,
                                                                                T_,
                                                                                N_
                                                                                )

                imp_solver_.workers.foreach_worker(
                            lambda ev: ev.foreach_env(
                                lambda env: env.set_task(imp_context_distribution_)))
                if update_round < 2:
                    for _ in range(5):
                        imp_solver_.train()
            context_curr_ = context_history_imp_[-1][:,which_param_]
            if context_ is None:
                context_ = context_curr_
                context_mean_ = [context_curr_.mean()]
            else:
                context_ = np.concatenate((context_, context_curr_))
                context_mean_ += [context_curr_.mean()]
            print("round", (i_ + 1, j_ + 1))
    context_mean_ = np.array(context_mean_)

    return context_, context_mean_

# predertimined params
N = 1000
T = 100
left_bound = np.ones((N,)) * 0.0
pow = np.ones((N,)) * 2
drift = np.random.normal(loc=0.00, scale=0.001, size=(N,))
step_size = np.ones((N,)) * 0.025
right_bound = np.random.normal(loc=2.5, scale=0.5, size=(N,))
noise = np.ones((N,)) * 0.05
context_particles = np.abs(np.vstack((left_bound, right_bound, pow, step_size, noise, drift)).T)

n_trajectory = 15
n_round = 3
which_param = 1

# train exact solver
c = {'context_distribution':
         ConstantDistribution(dim=6,
                              constant_vector=np.array([0.0, 2, 2, 0.05, 0.05, 0.0]))}

ray.shutdown()
ray.init()
expert = ppo.PPOTrainer(env=ContextualCliff, config={
    "env_config": c,
    "framework": "torch",  # config to pass to env class
})

rews = []
for eps in range(30):
    res = expert.train()
    if eps % 5 == 0:
        print(eps, res['episode_reward_mean'])
    rews += [res['episode_reward_mean']]

# train misspecified solver
c_mis = {'context_distribution':
             ConstantDistribution(dim=5,
                                  constant_vector=np.array([0.0, 1.5, 2.0, 0.025, 0.05, 0.0]))}

ray.shutdown()
ray.init()
mis_solver = ppo.PPOTrainer(env=ContextualCliff, config={
    "env_config": c_mis,
    "framework": "torch",  # config to pass to env class
})

for eps in range(30):
    res = mis_solver.train()
    if eps % 5 == 0:
        print(eps, res['episode_reward_mean'])

# train uniform solver
c_uniform = {'context_distribution':
             UniformDistribution(dim=6,
                              lower_bound_vector=np.array([0.0, 1.0, 2, 0.05, 0.05, 0.0]),
                              upper_bound_vector=np.array([0.0, 3.0, 2, 0.05, 0.05, 0.0]))}

ray.shutdown()
ray.init()
uniform_solver = ppo.PPOTrainer(env=ContextualCliff, config={
                                                    "env_config": c_uniform,
                                                    "framework": "torch",  # config to pass to env class
                                                })

rews = []
for eps in range(30):
    res = uniform_solver.train()
    if eps % 5 == 0:
        print(eps, res['episode_reward_mean'])
    rews += [res['episode_reward_mean']]

# context                    
context_exact = None
context_mean_exact = None
context_mis = None
context_mean_mis = None
context_uniform = None
context_mean_uniform = None
context_imp = None
context_mean_imp = None

for i in range(n_trajectory):

    for _ in range(10):
        gt_obs_arr, _ = get_rollouts(expert, config=c)
        if gt_obs_arr.shape[0] == 100:
            break
    assert gt_obs_arr.shape[0] == 100

    for j in range(n_round):

        # exact
        context_distribution = ParticleDistribution(
            dim=6, particles=context_particles, n_particles=N)
        context_history_exact, _ = filter_context(expert,
                                                context_distribution,
                                                gt_obs_arr,
                                                T,
                                                N
                                                )
        context_curr_exact = context_history_exact[-1][:, which_param]
        if context_exact is None:
            context_exact = context_curr_exact
            context_mean_exact = [context_curr_exact.mean()]
        else:
            context_exact = np.concatenate((context_exact, context_curr_exact))
            context_mean_exact += [context_curr_exact.mean()]
        print("exact", "round", (i + 1, j + 1))

        # mis
        context_distribution = ParticleDistribution(
            dim=6, particles=context_particles, n_particles=N)
        context_history_mis, _ = filter_context(mis_solver,
                                                context_distribution,
                                                gt_obs_arr,
                                                T,
                                                N
                                                )
        context_curr_mis = context_history_mis[-1][:, which_param]
        if context_mis is None:
            context_mis = context_curr_mis
            context_mean_mis = [context_curr_mis.mean()]
        else:
            context_mis = np.concatenate((context_mis, context_curr_mis))
            context_mean_mis += [context_curr_mis.mean()]
        print("mis", "round", (i + 1, j + 1))

        # uniform
        context_distribution = ParticleDistribution(
            dim=6, particles=context_particles, n_particles=N)
        context_history_uniform, _ = filter_context(uniform_solver,
                                                context_distribution,
                                                gt_obs_arr,
                                                T,
                                                N
                                                )
        context_curr_uniform = context_history_uniform[-1][:, which_param]
        if context_uniform is None:
            context_uniform = context_curr_uniform
            context_mean_uniform = [context_curr_uniform.mean()]
        else:
            context_uniform = np.concatenate((context_uniform, context_curr_uniform))
            context_mean_uniform += [context_curr_uniform.mean()]
        print("uniform", "round", (i + 1, j + 1))

        #imp
        imp_context_distribution = ParticleDistribution(dim=6, particles=context_particles, n_particles=N)

        ray.shutdown()
        ray.init()

        imp_solver = ppo.PPOTrainer(env=ContextualCliff, config={
                                                        "env_config":  {"context_distribution": imp_context_distribution},
                                                        "framework": "torch",
                                                    })

        for update_round in range(4):
            # burn in training
            if update_round == 0:
                for _ in range(15):
                    imp_solver.train()

            context_history_imp, imp_context_distribution = filter_context(imp_solver,
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

        context_curr_imp = context_history_imp[-1][:,which_param]
        if context_imp is None:
            context_imp = context_curr_imp
            context_mean_imp = [context_curr_imp.mean()]
        else:
            context_imp = np.concatenate((context_imp, context_curr_imp))
            context_mean_imp += [context_curr_imp.mean()]
        print("imp", "round", (i + 1, j + 1))

context_mean_exact = np.array(context_mean_exact)
context_mean_mis = np.array(context_mean_mis)
context_mean_uniform = np.array(context_mean_uniform)
context_mean_imp = np.array(context_mean_imp)

np.save("context_exact_cliff.npy", context_exact)
np.save("context_mean_exact_cliff.npy", context_mean_exact)
np.save("context_mis_cliff.npy", context_mis)
np.save("context_mean_mis_cliff.npy", context_mean_mis)
np.save("context_uniform_cliff.npy", context_uniform)
np.save("context_mean_uniform_cliff.npy", context_mean_uniform)
np.save("context_imp_cliff.npy", context_imp)
np.save("context_mean_imp_cliff.npy", context_mean_imp)