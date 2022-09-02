import numpy as np
from solver.value_iteration import VITrainer
import sys
import logging
import argparse

from mdps.whited import cMDPWhited
from utils.distributions import ConstantDistribution, ParticleDistribution, UniformDistribution

# parse command line arguments
parser = argparse.ArgumentParser(description="big big elephant")

parser.add_argument("-i", "--index", type=int,
                    required=True, help="index of working task")
parser.add_argument("-g", "--gt", type=int,
                    required=True, help="ten times of ground truth")
parser.add_argument("-n", "--node-name", type=str, required=False, default="unknown",
                    help="node name")  # TODO: get node name from env:SLURMD_NODENAME
parser.add_argument("-c", "--num-cpus", type=int, required=False,
                    choices=range(1, 8), default=2, help="number of CPUs allocated")

args = parser.parse_args()
index = args.index
gt = args.gt

# config logger
logger = logging.getLogger(__name__)
sys_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s [%(node_name)s] [%(index)3s] [%(gt)3s] [%(filename)15s:%(lineno)-4s] [%(levelname)-5s] %(message)s")
sys_handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(sys_handler)
logger = logging.LoggerAdapter(
    logger, {"index": args.index, "gt": args.gt, "node_name": args.node_name})
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
    return gt_obs_arr_[1:, :], gt_act_arr_, gt_rew_arr_


c = {"context_distribution":
     ConstantDistribution(dim=5, constant_vector=np.array(
         [0.98, 0.15, gt/10, 0.7, 0.15]))
     }
c_trainer = {'grid_nums':
             np.array([5, 15])
             }
c_sim = {"context_distribution":
         ConstantDistribution(dim=5, constant_vector=np.array(
             [0.98, 0.15, gt/10, 0.7, 0.15]))
         }
c_trainer_sim = {'grid_nums':
                 np.array([100, 20])
                 }
solver_vi = VITrainer(env=cMDPWhited(config=c),
                      config=c_trainer_sim)
logger.info("train solver")
solver_vi.train()


def filter_context(context_distribution_,
                   gt_obs_arr_,
                   T_,
                   N_,
                   c_trainer_
                   ):
    state_arr_ = np.ones((N_, 2))
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
            solver_ = VITrainer(env=env_,
                                config=c_trainer_)
            solver_.train()
            obs_ = env_.reset()
            if t_ > 0:
                env_.mdp.state = state_arr_[n_]
                obs_ = np.concatenate(
                    (np.array(env_.mdp.state), context_), axis=0).flatten()
            action_ = solver_.compute_single_action(obs_)
            obs_, reward_, _, _ = env_.step(action_)
            # estimate likelihood if r >= 1
            action_arr_[n_] = action_
            q = env_.likelihood(gt_obs_arr_[t_], action_arr_[
                                n_], obs_, reward_)
            qs_[n_] = q
            state_arr_[n_] = np.copy(env_.mdp.state)
            if n_ % 100 == 0:
                logger.debug("%s %s" % ("particle", n_))
        if t_ >= 1:
            # truncated importance sampling; [https://arxiv.org/pdf/1905.09800.pdf]
            qs_ = np.clip(qs_, 0, np.percentile(qs_, 90))
            if qs_.sum() == 0:
                continue
            qs_ = qs_ / qs_.sum()
            resample_index_ = context_distribution_.resample_particles_from_probability(
                p=qs_)
            p_temp_ = context_distribution_.particles
            p_noise_ = np.random.normal(
                loc=0, scale=p_temp_.std(axis=0), size=p_temp_.shape) * 0.1
            context_distribution_.particles += p_noise_
            context_distribution_.particles = np.clip(
                context_distribution_.particles, 0.0, 1.0)
            state_arr_ = state_arr_[resample_index_]
            action_arr_ = action_arr_[resample_index_]
            logger.debug("%s %s %s %s" % ("round", t_, "posterior mean",
                         context_distribution_.particles[:, 2].mean()))
        context_history_ += [context_distribution_.particles.copy()]
    return context_history_, context_distribution_


N = 1000
T = 100
which_param = 2

logger.info("simulate trajectory")
gt_obs_arr, gt_act_arr, gt_rew_arr = get_rollouts(solver_vi, config=c_sim)


gamma = np.ones((N,)) * 0.98
delta = np.ones((N,)) * 0.15
theta = np.random.uniform(0.4, 0.8, size=(N,))
rho = np.ones((N,)) * 0.7
sigma = np.ones((N,)) * 0.15

context_particles = np.abs(np.vstack((gamma, delta, theta, rho, sigma)).T)
context_distribution = ParticleDistribution(
    dim=5, particles=context_particles, n_particles=N)

logger.info("begin estimation")
context_history_vi, _ = filter_context(context_distribution,
                                       gt_obs_arr,
                                       T,
                                       N,
                                       c_trainer
                                       )
context = context_history_vi[-1][:, which_param]

logger.info("save results")
np.save("context_vi_"+str(index)+"_gt"+str(gt)+".npy", context)
