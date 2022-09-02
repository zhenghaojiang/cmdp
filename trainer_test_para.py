import numpy as np
import ray
from ray.rllib.agents import ppo
import sys
import logging
import argparse

from mdps.whited import cMDPWhited
from utils.distributions import ConstantDistribution, ParticleDistribution, UniformDistribution

# parse command line arguments
parser = argparse.ArgumentParser(description="big big elephant")

parser.add_argument("-i", "--index", type=int, required=True, help="index of working task")
parser.add_argument("-n", "--node-name", type=str, required=False, default="unknown", help="node name") # TODO: get node name from env:SLURMD_NODENAME
parser.add_argument("-c", "--num-cpus", type=int, required=False, choices=range(1, 8), default=2, help="number of CPUs allocated")

args = parser.parse_args()
index = args.index

# config logger
logger = logging.getLogger(__name__)
sys_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s [%(node_name)s] [%(index)3s] [%(filename)15s:%(lineno)-4s] [%(levelname)-5s] %(message)s")
sys_handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(sys_handler)
logger = logging.LoggerAdapter(logger, {"index": args.index, "node_name": args.node_name})
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

c = {"context_distribution":
        ConstantDistribution(dim=5, constant_vector=np.array([0.98, 0.15, 0.7, 0.7, 0.15]))
    }

logger.info("begin training")

ray.shutdown()
ray.init(num_cpus=args.num_cpus, include_dashboard=False, _system_config={"worker_register_timeout_seconds": 600})

config = {
    "env_config": c,
    "framework": "torch",  # config to pass to env class
    "num_workers": args.num_cpus - 1,
    "model": {
        # "fcnet_hiddens": [512, 512, 512],
        # "fcnet_activation": "swish",
    },
    # "lr": 0.00001,
    # "clip_param": 0.2, 
}

expert = ppo.PPOTrainer(env=cMDPWhited, config=config)

rews = []
for eps in range(50):
    res = expert.train()
    if eps % 5 == 0:
        logger.debug("eps: %s, res: %s" % (eps, res['episode_reward_mean']))
    rews += [res['episode_reward_mean']]

r = np.zeros(3000)
for i in range(3000):
    gt_obs_arr, gt_act_arr, gt_rew_arr = get_rollouts(expert, config=c)
    r[i] = sum(gt_rew_arr)

logger.info("save results")
np.save("trainer_test_"+str(index)+".npy", np.mean(r))