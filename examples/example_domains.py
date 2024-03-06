import gym
import numpy as np
from fosco.common import domains

import socnavgym
import fosco
from examples.example_seeding import SeedWrapper
from socnavgym.wrappers import WorldFrameObservations

"""
Convert observation in world frame to domains.
"""


def main():
    cfg = "../environment_configs/exp4_static.yaml"
    env = gym.make("SocNavGym-v1", config=cfg)
    seed = np.random.randint(1e6)
    max_steps = 100

    # obs: 6-d encoding, gx, gy, x, y, sintheta, costheta, velx, vely, vela, radius
    env = WorldFrameObservations(env)
    env = SeedWrapper(env=env, seed=seed)

    # extract dynamics params
    dt = env.TIMESTEP
    max_vx = env.MAX_ADVANCE_ROBOT
    max_vtheta = env.MAX_ROTATION

    obs, info = env.reset()
    done = False
    env.render()

    # create domains
    state_domain = None
    init_domain = None
    unsafe_domains = {}

    # state domain
    state_domain = domains.Rectangle(vars=["x", "y"], lb=(-env.MAP_X, -env.MAP_Y), ub=(env.MAP_X, env.MAP_Y))

    # init domain
    robot_obs = obs["robot"]
    robot_x = robot_obs[8]
    robot_y = robot_obs[9]
    robot_r = robot_obs[15]
    init_domain = domains.Sphere(vars=["x", "y"], centre=(robot_x, robot_y), radius=robot_r)

    for actor_group in obs:
        if actor_group in ["robot", "walls"]:
            continue

        print(actor_group)
        n_feats = 14
        n_actors = len(obs[actor_group]) // n_feats  # 13 features per actor
        group_domains = []
        for i in range(n_actors):
            actor_obs = obs[actor_group][i * n_feats:(i + 1) * n_feats]
            actor_x = actor_obs[6]
            actor_y = actor_obs[7]
            actor_r = actor_obs[10]
            actor_domain = domains.Sphere(vars=["x", "y"], centre=(actor_x, actor_y), radius=actor_r)
            group_domains.append(actor_domain)

        unsafe_domains[actor_group] = domains.Union(sets=group_domains)

    # visualize domains
    import matplotlib.pyplot as plt

    # plot initial domain
    all_points = state_domain.generate_data(batch_size=1000)
    #plt.scatter(all_points[:, 0], all_points[:, 1], label="state domain")

    init_points = init_domain.generate_data(batch_size=1000)
    plt.scatter(init_points[:, 0], init_points[:, 1], label="init domain - robot")

    for actor_group in list(unsafe_domains.keys())[::-1]:
        unsafe_dom = unsafe_domains[actor_group]
        unsafe_points = unsafe_dom.generate_data(batch_size=1000)
        plt.scatter(unsafe_points[:, 0], unsafe_points[:, 1], label=actor_group)
    plt.axis("equal")
    plt.legend()
    plt.show()


    # run the simulation
    step = 0
    while not done and step < max_steps:
        action = env.action_space.sample()
        new_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        env.render()
        input()



        obs = new_obs

    env.close()

if __name__=="__main__":
    main()