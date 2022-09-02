import numpy as np
import ray
from ray.rllib.agents import ppo

from mdps.whited_replace import cMDPWhited
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

c = {"context_distribution":
        ConstantDistribution(dim=5, constant_vector=np.array([0.98, 0.15, 0.7, 0.7, 0.15]))
    }

print("begin training")
ray.shutdown()
ray.init()

config = {
    "env_config": c,
    "framework": "torch",  # config to pass to env class
    "model": {
        # By default, the MODEL_DEFAULTS dict above will be used.

        # Change individual keys in that dict by overriding them, e.g.
        # "fcnet_hiddens": [512, 512, 512],
        # "fcnet_activation": "swish",
        # "use_lstm": True,
        # "use_attention": True,
    },
    # "lr": 0.00002,
    "clip_param": 0.2, 
}

expert = ppo.PPOTrainer(env=cMDPWhited, config=config)

rews = []
for eps in range(50):
    res = expert.train()
    if eps % 5 == 0:
        print(eps, res['episode_reward_mean'])
    rews += [res['episode_reward_mean']]
print("end training")

r = np.zeros(3000)
for i in range(3000):
    gt_obs_arr, gt_act_arr, gt_rew_arr = get_rollouts(expert, config=c)
    r[i] = sum(gt_rew_arr)
print("mean reward")
print(np.mean(r))
