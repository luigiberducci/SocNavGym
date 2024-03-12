import gym
import numpy as np
import torch
from fosco.systems.system_env import TensorType
from fosco.verifier.types import DRSYMBOL

import socnavgym
from examples.example_seeding import SeedWrapper
from socnavgym.wrappers import WorldFrameObservations
from fosco.common.functions import FUNCTIONS

"""
The robot dynamics can be either holonomic or diff-drive.
For diff-drive:
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

For holonomic:
    state = [x, y, theta] where x, y denote the position, theta the orientation in global frame
    control = [vx, vy, vtheta] where in addition to vx, vtheta, we explicitely control vy
    
    dx/dt = vx cos (theta)
    dy/dt = vx sin (theta)
    dtheta/dt = vtheta
    
    in the dynamics there is also a term dependent on vy, but for non-holonomic robot, this is set to 0.
    
    note: the action space is defined in +-1 for vx, vtheta
    the actions are then transformed to [min,max] value of each domain
    - vx in [-MAX_ADVANCE_ROBOT, MAX_ADVANCE_ROBOT]
    - vtheta in [-MAX_ROTATION, MAX_ROTATION]
"""


def fx_torch(x: TensorType) -> TensorType:
    assert (
        len(x.shape) == 3
    ), f"expected batched input with shape (batch_size, state_dim, 1), got shape {x.shape}"
    if isinstance(x, np.ndarray):
        fx = np.zeros_like(x)
    else:
        fx = torch.zeros_like(x)
    return fx


def fx_smt(x: list) -> np.ndarray | torch.Tensor:
    assert isinstance(
        x, list
    ), "expected list of symbolic state variables, [x0, x1, ...]"
    return np.zeros(len(x))


def gxdd_torch(x: TensorType) -> TensorType:
    """
    g(x) for diff-drive model
    """
    assert (
        len(x.shape) == 3
    ), "expected batched input with shape (batch_size, state_dim, 1)"
    if isinstance(x, np.ndarray):
        cosx = np.cos(x[:, 2, :])
        sinx = np.sin(x[:, 2, :])
        # make a batch of matrices, each with
        # [[cos(theta_i), 0][sin(theta_), 0][0, 1]]
        gx = np.array(
            [[[cosx[i][0], 0], [sinx[i][0], 0], [0, 1]] for i in range(x.shape[0])]
        )
    else:
        cosx = torch.cos(x[:, 2, :])
        sinx = torch.sin(x[:, 2, :])

        gx = torch.tensor(
            [[[cosx[i][0], 0], [sinx[i][0], 0], [0, 1]] for i in range(x.shape[0])]
        )
    return gx


def gxho_torch(x: TensorType) -> TensorType:
    """
    g(x) for holonomic model
    """
    assert (
        len(x.shape) == 3
    ), "expected batched input with shape (batch_size, state_dim, 1)"
    if isinstance(x, np.ndarray):
        costheta = np.cos(x[:, 2, :])
        sintheta = np.sin(x[:, 2, :])
        costheta_pi2 = np.cos(x[:, 2, :] + np.pi / 2)
        sintheta_pi2 = np.sin(x[:, 2, :] + np.pi / 2)

        gx = np.array(
            [
                [
                    [costheta[i][0], costheta_pi2[i][0], 0.0],
                    [sintheta[i][0], sintheta_pi2[i][0], 0.0],
                    [0.0, 0.0, 1.0],
                ]
                for i in range(x.shape[0])
            ]
        )
    else:
        costheta = torch.cos(x[:, 2, :])
        sintheta = torch.sin(x[:, 2, :])
        costheta_pi2 = torch.cos(x[:, 2, :] + torch.pi / 2)
        sintheta_pi2 = torch.sin(x[:, 2, :] + torch.pi / 2)

        gx = torch.tensor(
            [
                [
                    [costheta[i][0], costheta_pi2[i][0], 0.0],
                    [sintheta[i][0], sintheta_pi2[i][0], 0.0],
                    [0.0, 0.0, 1.0],
                ]
                for i in range(x.shape[0])
            ]
        )
    return gx


def gxdd_smt(x: list) -> TensorType:
    assert isinstance(
        x, list
    ), "expected list of symbolic state variables, [x0, x1, ...]"
    assert all(
        [isinstance(xi, DRSYMBOL) for xi in x]
    ), f"expected list of dreal variables, got {x}"
    fns = FUNCTIONS["dreal"]
    Sin_ = fns["Sin"]
    Cos_ = fns["Cos"]

    return np.array([[Cos_(x[2]), 0.0], [Sin_(x[2]), 0.0], [0.0, 1.0]])


def gxho_smt(x: list) -> TensorType:
    assert isinstance(
        x, list
    ), "expected list of symbolic state variables, [x0, x1, ...]"
    assert all(
        [isinstance(xi, DRSYMBOL) for xi in x]
    ), f"expected list of dreal variables, got {x}"
    fns = FUNCTIONS["dreal"]
    Sin_ = fns["Sin"]
    Cos_ = fns["Cos"]

    return np.array(
        [
            [Cos_(x[2]), Cos_(x[2] + np.pi / 2), 0.0],
            [Sin_(x[2]), Sin_(x[2] + np.pi / 2), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def dynamics(x: np.ndarray, u: np.ndarray, dt: float, holonomic: bool) -> np.ndarray:
    assert isinstance(x, np.ndarray) and x.shape == (3,), f"wrong x type: got {x}"
    assert isinstance(dt, float | int) and dt > 0, f"wrong dt, got {dt}"
    x = x[None].reshape(1, -1, 1)

    if holonomic:
        assert isinstance(u, np.ndarray) and u.shape == (3,), f"wrong u type: got {u}"
        u = u[None].reshape(1, -1, 1)
        dxdt = fx_torch(x) + gxho_torch(x) @ u
    else:
        assert isinstance(u, np.ndarray) and u.shape == (2,), f"wrong u type: got {u}"
        u = u[None].reshape(1, -1, 1)
        dxdt = fx_torch(x) + gxdd_torch(x) @ u

    next_x = x + dt * dxdt
    return next_x.squeeze()


def obs_to_state(obs: dict[str, np.array], agent_id: str = "robot") -> np.ndarray:
    """
    Convert the multi-agent observation to a state vector for the given agent.
    """
    agent_obd = obs[agent_id]
    assert isinstance(agent_obd, np.ndarray) and agent_obd.shape == (16,)

    ids = [
        "enc0",
        "enc1",
        "enc2",
        "enc3",
        "enc4",
        "enc5",
        "gx",
        "gy",
        "x",
        "y",
        "sintheta",
        "costheta",
        "vx",
        "vy",
        "vtheta",
        "radius",
    ]

    x_id = ids.index("x")
    y_id = ids.index("y")
    sint_id = ids.index("sintheta")
    cost_id = ids.index("costheta")

    x = agent_obd[x_id]
    y = agent_obd[y_id]
    theta = np.arctan2(agent_obd[sint_id], agent_obd[cost_id])

    state = np.array([x, y, theta], dtype=np.float32)
    assert isinstance(state, np.ndarray) and state.shape == (
        3,
    ), f"wrong state, expected 3d, got {state}"
    return state


def action_to_control(
    action: np.ndarray, maxvx: float, maxvy: float, maxtheta: float, holonomic: bool
) -> np.ndarray:
    assert isinstance(action, np.ndarray) and action.shape == (3,)
    assert all([-1.0 < a <= 1 for a in action])
    if holonomic:
        u = np.array(
            [action[0] * maxvx, action[1] * maxvy, action[2] * maxtheta],
            dtype=np.float32,
        )
    else:
        u = np.array([action[0] * maxvx, action[2] * maxtheta], dtype=np.float32)
    return u


def main():
    cfg = "../environment_configs/exp4_static.yaml"
    env = gym.make("SocNavGym-v1", config=cfg)
    seed = np.random.randint(1e6)
    max_steps = 100

    # obs: 6-d encoding, gx, gy, x, y, sintheta, costheta, velx, vely, vela, radius
    env = WorldFrameObservations(env)
    env = SeedWrapper(env=env, seed=seed)

    # extract dynamics params
    is_holonomic = env.robot.type == "holonomic"
    dt = env.TIMESTEP
    max_vx = env.MAX_ADVANCE_ROBOT
    max_vy = env.MAX_ADVANCE_ROBOT
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
        ctrl = action_to_control(
            action=action,
            maxvx=max_vx,
            maxvy=max_vy,
            maxtheta=max_vtheta,
            holonomic=is_holonomic,
        )
        pred_state = dynamics(
            x=state, u=ctrl, dt=dt, holonomic=is_holonomic
        )  # from my dynamics

        # orientation in +-np.pi
        pred_state[2] = (pred_state[2] + np.pi) % (2 * np.pi) - np.pi

        new_state = obs_to_state(new_obs, agent_id="robot")  # from sim
        assert np.allclose(
            pred_state, new_state
        ), f"got pred={pred_state}, new={new_state}"

        obs = new_obs

    env.close()


if __name__ == "__main__":
    main()
