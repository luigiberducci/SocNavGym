import gymnasium as gym
import socnavgym


def main():
    cfg = "../environment_configs/exp4_no_sngnn.yaml"
    env = gym.make("SocNavGym-v1", config=cfg, render_mode="human")

    for _ in range(10):
        obs, info = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            env.render()

        #input()

    env.close()


if __name__ == "__main__":
    main()
