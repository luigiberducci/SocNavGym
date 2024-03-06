import gym
import numpy as np

import socnavgym
from examples.example_seeding import SeedWrapper
from socnavgym.wrappers import WorldFrameObservations

"""
The robot dynamics are quite easy
state = [x, y, theta] where x, y denote the position, theta the orientation in global frame
control = [vx, vtheta] where vx is the forward velocity in the robot frame, vtheta is the angular velocity

dx/dt = vx cos (theta)
dy/dt = vx sin (theta)
dtheta/dt = vtheta

in the dynamics there is also a term dependent on vy, but for non-holonomic robot, this is set to 0.

note: the action space is defined in +-1 for vx, vtheta
the actions are then transformed to [min,max] value of each domain
- vx in [-MAX_ADVANCE_ROBOT, MAX_ADVANCE_ROBOT]
- vtheta in [-MAX_ROTATION, MAX_ROTATION]

"""

def dynamics(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    assert isinstance(x, np.ndarray) and x.shape == (3,), f"wrong x type: got {x}"
    assert isinstance(u, np.ndarray) and u.shape == (2,), f"wrong u type: got {x}"
    assert isinstance(dt, float | int) and dt > 0, f"wrong dt, got {dt}"
    dxdt = np.array([
        u[0] * np.cos(x[2]),
        u[0] * np.sin(x[2]),
        u[1],
    ], dtype=np.float32)
    return x + dt * dxdt

def obs_to_state(obs: dict[str, np.array], agent_id: str = "robot") -> np.ndarray:
    """
    Convert the multi-agent observation to a state vector for the given agent.
    """
    agent_obd = obs[agent_id]
    assert isinstance(agent_obd, np.ndarray) and agent_obd.shape == (16,)

    ids = ["enc0", "enc1", "enc2", "enc3", "enc4", "enc5",
           "gx", "gy", "x", "y", "sintheta", "costheta",
           "vx", "vy", "vtheta", "radius"]

    x_id = ids.index("x")
    y_id = ids.index("y")
    sint_id = ids.index("sintheta")
    cost_id = ids.index("costheta")

    x = agent_obd[x_id]
    y = agent_obd[y_id]
    theta = np.arctan2(agent_obd[sint_id], agent_obd[cost_id])

    state = np.array([x, y, theta], dtype=np.float32)
    assert isinstance(state, np.ndarray) and state.shape == (3,), f"wrong state, expected 3d, got {state}"
    return state

def action_to_control(action: np.ndarray, maxvx: float, maxtheta: float) -> np.ndarray:
    assert isinstance(action, np.ndarray) and action.shape == (3,)
    assert all([-1.0 < a <= 1 for a in action])
    u = np.array([action[0] * maxvx,
                  action[2] * maxtheta], dtype=np.float32)
    assert isinstance(u, np.ndarray) and u.shape == (2,)
    return u

def main():
    cfg = "../environment_configs/exp4_static.yaml"
    env = gym.make("SocNavGym-v1", config=cfg)
    seed = 533 # np.random.randint(1e6)
    max_steps = 100

    # obs: 6-d encoding, gx, gy, x, y, sintheta, costheta, velx, vely, vela, radius
    idx, idy, idst, idct = 8, 9, 10, 11
    env = WorldFrameObservations(env)
    env = SeedWrapper(env=env, seed=seed)

    # extract dynamics params
    dt = env.TIMESTEP
    max_vx = env.MAX_ADVANCE_ROBOT
    max_vtheta = env.MAX_ROTATION

    obs, info = env.reset()
    done = False
    env.render()

    step = 0
    while not done and step < max_steps:
        action = env.action_space.sample()
        new_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        env.render()

        # my prediction
        state = obs_to_state(obs, agent_id="robot")
        ctrl = action_to_control(action=action, maxvx=max_vx, maxtheta=max_vtheta)
        pred_state = dynamics(x=state, u=ctrl, dt=dt)   # from my dynamics
        new_state = obs_to_state(new_obs, agent_id="robot") # from sim
        assert np.allclose(pred_state, new_state)

        obs = new_obs

    env.close()

if __name__=="__main__":
    main()