import gym
import socnavgym

import torch
from socnavgym.wrappers import DiscreteActions
from stable_baselines3 import DQN
import argparse




ap = argparse.ArgumentParser()
ap.add_argument("-e", "--env_config", help="path to environment config", required=True)
ap.add_argument("-s", "--save_path", help="path to save the model", required=True)
ap.add_argument("-d", "--use_deep_net", help="True or False, based on whether you want a transformer based feature extractor", required=False, default=False)
ap.add_argument("-g", "--gpu", help="gpu id to use", required=False, default="0")
args = vars(ap.parse_args())

env = gym.make("SocNavGym-v1", config=args["env_config"])
env = DiscreteActions(env)

net_arch = {}

if not args["use_deep_net"]:
    net_arch = [512, 256, 128, 64]

else:
    net_arch = [512, 256, 256, 256, 128, 128, 64]

policy_kwargs = {"net_arch" : net_arch}

device = 'cuda:'+str(args["gpu"]) if torch.cuda.is_available() else 'cpu'
model = DQN("MultiInputPolicy", env, verbose=0, policy_kwargs=policy_kwargs, device=device)
model.learn(total_timesteps=50000*200)
model.save(args["save_path"])