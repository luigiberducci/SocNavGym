import warnings

import gymnasium as gym
import numpy as np
from fosco.common import domains


from examples.example_seeding import SeedWrapper
from socnavgym.envs import SocNavEnv
from socnavgym.wrappers import WorldFrameObservations

"""
Convert observation in world frame to domains.
"""

def get_state_domain(env: SocNavEnv) -> domains.Set:
    if env.robot.type in ["holonomic", "diff-drive"]:
        dom = domains.Rectangle(
            vars=["x", "y", "theta"], lb=(-env.MAP_X, -env.MAP_Y, -np.pi), ub=(env.MAP_X, env.MAP_Y, np.pi)
        )
    else:
        dom = domains.Rectangle(
            vars=["x", "y"], lb=(-env.MAP_X, -env.MAP_Y), ub=(env.MAP_X, env.MAP_Y)
        )
    return dom
def get_input_domain(env: SocNavEnv) -> domains.Set:
    max_vx = env.MAX_ADVANCE_ROBOT
    max_vy = env.MAX_ADVANCE_ROBOT
    max_vtheta = env.MAX_ROTATION

    if env.robot.type == "holonomic":
        domain = domains.Rectangle(
            vars=["vx", "vy", "vtheta"], lb=(-max_vx, -max_vy, -max_vtheta), ub=(max_vx, max_vy, max_vtheta)
        )
    elif env.robot.type == "diff-drive":
        domain = domains.Rectangle(
            vars=["vx", "vtheta"], lb=(-max_vx, -max_vtheta), ub=(max_vx, max_vtheta)
        )
    elif env.robot.type == "integrator":
        domain = domains.Rectangle(
            vars=["vx", "vy"], lb=(-max_vx, -max_vy), ub=(max_vx, max_vy)
        )
    return domain

def get_init_domain(env: SocNavEnv, use_only_boxes: bool = False) -> domains.Set:
    if not isinstance(env, SeedWrapper):
        raise TypeError("init domain relies on having a seeded environment")

    obs, info = env.reset()

    robot_obs = obs["robot"]
    robot_x = robot_obs[8]
    robot_y = robot_obs[9]
    robot_r = robot_obs[15]

    if use_only_boxes:
        if env.robot.type in ["holonomic", "diff-drive"]:
            lowerbound = np.array([robot_x, robot_y, 0.0]) - np.array([robot_r, robot_r, np.pi])
            upperbound = np.array([robot_x, robot_y, 0.0]) + np.array([robot_r, robot_r, np.pi])
            actor_domain = domains.Rectangle(
                vars=["x", "y", "theta"], lb=lowerbound, ub=upperbound
            )
        else:
            lowerbound = np.array([robot_x, robot_y]) - np.array([robot_r, robot_r])
            upperbound = np.array([robot_x, robot_y]) + np.array([robot_r, robot_r])
            actor_domain = domains.Rectangle(
                vars=["x", "y"], lb=lowerbound, ub=upperbound
            )
    else:
        warnings.warn("using a sphere restricts theta by the robot radius, which is not correct")
        if env.robot.type in ["holonomic", "diff-drive"]:
            actor_domain = domains.Sphere(
                vars=["x", "y", "theta"], center=(robot_x, robot_y, 0.0), radius=robot_r
            )
        else:
            actor_domain = domains.Sphere(
                vars=["x", "y"], center=(robot_x, robot_y), radius=robot_r
            )


    return actor_domain

def get_unsafe_domain(env: SocNavEnv, use_only_boxes: bool = False) -> domains.Set:
    if not isinstance(env, SeedWrapper):
        raise TypeError("unsafe domain relies on having a seeded environment")

    obs, info = env.reset()

    unsafe_domains = {}
    for actor_group in obs:
        if actor_group in ["robot", "walls"]:
            continue

        n_feats = 14
        n_actors = len(obs[actor_group]) // n_feats  # 13 features per actor
        group_domains = []
        for i in range(n_actors):
            actor_obs = obs[actor_group][i * n_feats : (i + 1) * n_feats]
            actor_x = actor_obs[6]
            actor_y = actor_obs[7]
            actor_r = actor_obs[10]

            if actor_group == "tables":
                table_w = env.TABLE_WIDTH
                table_l = env.TABLE_LENGTH
                lowerbound = np.array([actor_x, actor_y, 0.0]) - np.array([table_l / 2, table_w / 2, np.pi])
                upperbound = np.array([actor_x, actor_y, 0.0]) + np.array([table_l / 2, table_w / 2, np.pi])
                if env.robot.type in ["holonomic", "diff-drive"]:
                    actor_domain = domains.Rectangle(
                        vars=["x", "y", "theta"], lb=lowerbound, ub=upperbound
                    )
                else:
                    actor_domain = domains.Rectangle(
                        vars=["x", "y"], lb=lowerbound[:2], ub=upperbound[:2]
                    )
            else:
                if use_only_boxes:
                    if env.robot.type in ["holonomic", "diff-drive"]:
                        lowerbound = np.array([actor_x, actor_y, 0.0]) - np.array([actor_r, actor_r, np.pi])
                        upperbound = np.array([actor_x, actor_y, 0.0]) + np.array([actor_r, actor_r, np.pi])
                        actor_domain = domains.Rectangle(
                            vars=["x", "y", "theta"], lb=lowerbound, ub=upperbound
                        )
                    else:
                        lowerbound = np.array([actor_x, actor_y]) - np.array([actor_r, actor_r])
                        upperbound = np.array([actor_x, actor_y]) + np.array([actor_r, actor_r])
                        actor_domain = domains.Rectangle(
                            vars=["x", "y"], lb=lowerbound, ub=upperbound
                        )
                else:
                    warnings.warn("using a sphere restricts theta by the robot radius, which is not correct")
                    if env.robot.type in ["holonomic", "diff-drive"]:
                        actor_domain = domains.Sphere(
                            vars=["x", "y", "theta"], center=(actor_x, actor_y, 0.0), radius=actor_r
                        )
                    else:
                        actor_domain = domains.Sphere(
                            vars=["x", "y"], center=(actor_x, actor_y), radius=actor_r
                        )

            group_domains.append(actor_domain)

        unsafe_domains[actor_group] = domains.Union(sets=group_domains)

    return domains.Union(sets=list(unsafe_domains.values()))
def main():
    cfg = "../environment_configs/exp4_static.yaml"  # static.yaml"
    env = gym.make("SocNavGym-v1", config=cfg)
    seed = np.random.randint(1e6)
    max_steps = 100
    use_only_boxes = True   # toggle using only boxes in domain abstraction

    # obs: 6-d encoding, gx, gy, x, y, sintheta, costheta, velx, vely, vela, radius
    env = WorldFrameObservations(env)
    env = SeedWrapper(env=env, seed=seed)
    print("seed:", seed)

    obs, info = env.reset()
    done = False
    env.render()


    # domains
    state_domain = get_state_domain(env=env)
    input_domain = get_input_domain(env=env)
    init_domain = get_init_domain(env=env, use_only_boxes=use_only_boxes)
    unsafe_domain = get_unsafe_domain(env=env, use_only_boxes=use_only_boxes)

    # print domains
    for k, dom in zip(["state", "input", "init", "unsafe"],
                      [state_domain, input_domain, init_domain, unsafe_domain]):
        print(f"{k} domain", dom)


    # visualize domains
    import matplotlib.pyplot as plt

    init_points = init_domain.generate_data(batch_size=1000)
    plt.scatter(init_points[:, 0], init_points[:, 1], label="init domain - robot")

    for unsafe_dom in unsafe_domain.sets[::-1]:
        unsafe_points = unsafe_dom.generate_data(batch_size=1000)
        plt.scatter(unsafe_points[:, 0], unsafe_points[:, 1])
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
        obs = new_obs

    env.close()


if __name__ == "__main__":
    main()
