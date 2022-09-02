import numpy as np
import sys
import logging
import argparse
import ray
from ray.rllib.agents import ppo

from mdps.whited import cMDPWhited
from utils.distributions import ConstantDistribution, ParticleDistribution, UniformDistribution
from estimators.whited_estimator import EstWhited

# parse command line arguments
parser = argparse.ArgumentParser(description="big big elephant")

parser.add_argument("-i", "--index", type=int, required=True, help="index of working task")
parser.add_argument("-s", "--sample-size", type=int, required=True, help="sample size")
parser.add_argument("-n", "--node-name", type=str, required=False, default="unknown", help="node name") # TODO: get node name from env:SLURMD_NODENAME
parser.add_argument("-c", "--num-cpus", type=int, required=False, choices=range(1, 8), default=2, help="number of CPUs allocated")

args = parser.parse_args()
index = args.index
sample_size = args.sample_size

# config logger
logger = logging.getLogger(__name__)
sys_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s [%(node_name)s] [%(index)3s] [%(sample_size)4s] [%(filename)15s:%(lineno)-4s] [%(levelname)-5s] %(message)s")
sys_handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(sys_handler)
logger = logging.LoggerAdapter(logger, {"index": args.index, "sample_size": args.sample_size, "node_name": args.node_name})
logger.info("init logger")

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
logger.info("exact solver")
c = {"context_distribution":
        ConstantDistribution(dim=5, constant_vector=np.array([0.98, 0.15, 0.7, 0.7, 0.15]))
    }

ray.shutdown()
ray.init(num_cpus=args.num_cpus, include_dashboard=False, _system_config={"worker_register_timeout_seconds": 600})

expert = ppo.PPOTrainer(env=cMDPWhited, config={
    "env_config": c,
    "framework": "torch",  # config to pass to env class
    "num_workers": args.num_cpus - 1,
})

rews = []
for eps in range(35):
    res = expert.train()
    if eps % 5 == 0:
        logger.debug("eps: %s, res: %s" % (eps, res['episode_reward_mean']))
    rews += [res['episode_reward_mean']]

# misspecified solver
logger.info("misspecified solver")
c_mis = {"context_distribution":
        ConstantDistribution(dim=5, constant_vector=np.array([0.98, 0.15, 0.5, 0.7, 0.15]))
    }

ray.shutdown()
ray.init(num_cpus=args.num_cpus, include_dashboard=False, _system_config={"worker_register_timeout_seconds": 600})

solver_mis = ppo.PPOTrainer(env=cMDPWhited, config={
    "env_config": c_mis,
    "framework": "torch",  # config to pass to env class
    "num_workers": args.num_cpus - 1,
})

rews = []
for eps in range(35):
    res = solver_mis.train()
    if eps % 5 == 0:
        logger.debug("eps: %s, res: %s" % (eps, res['episode_reward_mean']))
    rews += [res['episode_reward_mean']]

# uniform solver
logger.info("uniform solver")
c_uniform = {'context_distribution':
             UniformDistribution(dim=5,
                                 lower_bound_vector=np.array([0.98, 0.15, 0.4, 0.7, 0.15]),
                                 upper_bound_vector=np.array([0.98, 0.15, 0.8, 0.7, 0.15]))}

ray.shutdown()
ray.init(num_cpus=args.num_cpus, include_dashboard=False, _system_config={"worker_register_timeout_seconds": 600})
solver_uniform = ppo.PPOTrainer(env=cMDPWhited, config={
                                                    "env_config": c_uniform,
                                                    "framework": "torch",  # config to pass to env class
                                                    "num_workers": args.num_cpus - 1,
                                                })

rews = []
for eps in range(35):
    res = solver_uniform.train()
    if eps % 5 == 0:
        logger.debug("eps: %s, res: %s" % (eps, res['episode_reward_mean']))
    rews += [res['episode_reward_mean']]

# Pre-determined params
gt_obs_arr, _, _ = get_rollouts(expert, config=c)

N = sample_size

gamma = np.ones((N,)) * 0.98
delta = np.ones((N,)) * 0.15
theta = np.random.uniform(0.4, 0.8, size=(N,))
rho = np.ones((N,)) * 0.7
sigma = np.ones((N,)) * 0.15

# exact est

context_particles = np.abs(np.vstack((gamma, delta, theta, rho, sigma)).T)
context_distribution = ParticleDistribution(dim=5, particles=context_particles, n_particles=N)

est_config = {
    'solver': expert, 
    'context_distribution': context_distribution, 
    'observation_array': gt_obs_arr,
    'estimate_index': 2, 
    'particle_number': sample_size, 
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
    'estimate_index': 2, 
    'particle_number': sample_size, 
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
    'estimate_index': 2, 
    'particle_number': sample_size, 
}

estimator = EstWhited(config=est_config)
uniform, uniform_mean = estimator.estiamte()

# imp est
context_particles = np.abs(np.vstack((gamma, delta, theta, rho, sigma)).T)
imp_context_distribution = ParticleDistribution(dim=5, particles=context_particles, n_particles=N)

ray.shutdown()
ray.init(num_cpus=args.num_cpus, include_dashboard=False, _system_config={"worker_register_timeout_seconds": 600})

imp_solver = ppo.PPOTrainer(env=cMDPWhited, config={
                                                "env_config":  {"context_distribution": imp_context_distribution},
                                                "framework": "torch",
                                                "num_workers": args.num_cpus - 1,
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
        'estimate_index': 2, 
        'particle_number': sample_size, 
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
context_mean_exact = np.array(exact_mean)
context_mean_mis = np.array(mis_mean)
context_mean_uniform = np.array(uniform_mean)
context_mean_imp = np.array(imp_mean)

logger.info("save results")
np.save("context_exact_"+str(est_config['estimate_index'])+"_"+str(N)+"_"+str(index)+".npy", exact)
np.save("context_mean_exact_"+str(est_config['estimate_index'])+"_"+str(N)+"_"+str(index)+".npy", exact_mean)
np.save("context_mis_"+str(est_config['estimate_index'])+"_"+str(N)+"_"+str(index)+".npy", mis)
np.save("context_mean_mis_"+str(est_config['estimate_index'])+"_"+str(N)+"_"+str(index)+".npy", mis_mean)
np.save("context_uniform_"+str(est_config['estimate_index'])+"_"+str(N)+"_"+str(index)+".npy", uniform)
np.save("context_mean_uniform_"+str(est_config['estimate_index'])+"_"+str(N)+"_"+str(index)+".npy", uniform_mean)
np.save("context_imp_"+str(est_config['estimate_index'])+"_"+str(N)+"_"+str(index)+".npy", imp)
np.save("context_mean_imp_"+str(est_config['estimate_index'])+"_"+str(N)+"_"+str(index)+".npy", imp_mean)







