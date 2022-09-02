import numpy as np
from solver.value_iteration import VITrainer
import sys
import logging
import argparse

from mdps.whited import cMDPWhited
from utils.distributions import ConstantDistribution, ParticleDistribution, UniformDistribution

# parse command line arguments
parser = argparse.ArgumentParser(description="big big elephant")

parser.add_argument("-k", "--k-grid", type=int,
                    required=True, help="number of grids for k")
parser.add_argument("-z", "--z-grid", type=int,
                    required=True, help="number of grids for z")
parser.add_argument("-t", "--accuracy", type=int,
                    required=True, help="")
parser.add_argument("-n", "--node-name", type=str, required=False, default="unknown",
                    help="node name")  # TODO: get node name from env:SLURMD_NODENAME
parser.add_argument("-c", "--num-cpus", type=int, required=False,
                    choices=range(1, 8), default=2, help="number of CPUs allocated")

args = parser.parse_args()
k_grid = args.k_grid
z_grid = args.z_grid
ccrit = args.accuracy * 0.1

# config logger
logger = logging.getLogger(__name__)
sys_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s [%(node_name)s] [%(k_grid)s] [%(z_grid)s] [%(accuracy)s] [%(filename)15s:%(lineno)-4s] [%(levelname)-5s] %(message)s")
sys_handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(sys_handler)
logger = logging.LoggerAdapter(
    logger, {"k_grid": args.k_grid, "z_grid": args.z_grid, "accuracy": args.accuracy, "node_name": args.node_name})
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
         [0.98, 0.15, 0.7, 0.7, 0.15]))
     }
c_trainer = {'grid_nums':
             np.array([k_grid, z_grid]), 
             'ccrit': ccrit
             }
c_sim = {"context_distribution":
         ConstantDistribution(dim=5, constant_vector=np.array(
             [0.98, 0.15, 0.7, 0.7, 0.15]))
         }
solver_vi = VITrainer(env=cMDPWhited(config=c),
                      config=c_trainer)
logger.info("train solver")
solver_vi.train()

logger.info("calc reward")
r = np.zeros(5000)
for i in range(5000):
    gt_obs_arr, gt_act_arr, gt_rew_arr = get_rollouts(solver_vi, config=c_sim)
    r[i] = sum(gt_rew_arr)

logger.info("save results")
np.save("trainer_vi_"+str(k_grid)+"_"+str(z_grid)+"_"+str(ccrit)+"_"+".npy", np.mean(r))