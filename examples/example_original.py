from typing import Tuple

import gym
import numpy as np
from gym.core import ObsType

import socnavgym



class SeedWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env=env)
        self._seed = seed
    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        return super().reset(seed=self._seed)


def main():
    cfg = "../environment_configs/exp4_no_sngnn.yaml"
    env = gym.make("SocNavGym-v1", config=cfg)
    env = SeedWrapper(env=env, seed=533)

    for _ in range(10):
        obs, info = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            env.render()

        input()

    env.close()

if __name__=="__main__":
    main()