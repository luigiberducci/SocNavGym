from collections import namedtuple
from functools import partial
from typing import Callable

import torch
from fosco.systems.system_env import SystemEnv

import socnavgym
import gym

from examples.example_domains import (
    get_state_domain,
    get_init_domain,
    get_unsafe_domain,
    get_input_domain,
)
from examples.example_dynamics import fx_torch, fx_smt, gxdd_torch, gxdd_smt, gxho_torch, gxho_smt, gxint_torch, gxint_smt
from examples.example_seeding import SeedWrapper
from socnavgym.envs import SocNavEnv
from socnavgym.wrappers import WorldFrameObservations

from fosco.systems.system_from_spec import System
from fosco.config import CegisConfig
from fosco.common.consts import CertificateType, VerifierType, LossReLUType
from fosco.logger import LoggerType
from fosco.cegis import Cegis

from rl_trainer.run_ppo import Args as ppo_args
from rl_trainer.run_ppo import run

Abstraction = namedtuple("Abstraction", "variables controls dynamics domains")


def create_env_abstraction(env: SocNavEnv) -> Abstraction:

    if env.robot.type == "holonomic":
        variables = ["x", "y", "theta"]
        controls = ["vx", "vy", "vtheta"]
    elif env.robot.type == "diff-drive":
        variables = ["x", "y", "theta"]
        controls = ["vx", "vtheta"]
    elif env.robot.type == "integrator":
        variables = ["x", "y"]
        controls = ["vx", "vy"]

    domains = {
        "input": get_input_domain(env),
        "init": get_init_domain(env, use_only_boxes=True),
        "unsafe": get_unsafe_domain(env, use_only_boxes=True),
        "lie": get_state_domain(env),
    }

    gx_torch_dict = {"holonomic": gxho_torch, "diff-drive": gxdd_torch, "integrator": gxint_torch}
    gx_smt_dict = {"holonomic": gxho_smt, "diff-drive": gxdd_smt, "integrator": gxint_smt}
    gx_torch = gx_torch_dict[env.robot.type]
    gx_smt = gx_smt_dict[env.robot.type]
    dynamics = {
        "fx_torch": fx_torch,
        "fx_smt": fx_smt,
        "gx_torch": gx_torch,
        "gx_smt": gx_smt,
    }

    return Abstraction(
        variables=variables, controls=controls, dynamics=dynamics, domains=domains
    )


def create_data_generator(system: System) -> dict[str, Callable]:
    sets = system.domains
    return {
        "init": lambda n: sets["init"].generate_data(n),
        "unsafe": lambda n: sets["unsafe"].generate_data(n),
        "lie": lambda n: torch.concatenate(
            [sets["lie"].generate_data(n), sets["input"].generate_data(n)], dim=1
        ),
    }


def main(args):
    env_id = "SocNavGym-v1"
    env_cfg = "environment_configs/exp4_static.yaml"
    seed = 29397

    activations = ("sigmoid", "linear")
    n_hidden_neurons = (5, 5)
    n_data_samples = 2000
    max_cegis_iters = 50
    n_cegis_epochs = 100
    verifier_n_cex = 100
    verifier_timeout_s = 60
    learning_rate = 1e-3
    weight_decay = 1e-4
    loss_relu = LossReLUType.SOFTPLUS
    verbose = 1

    #
    exp_name = f"{env_id}_{seed}"

    # create environment
    def make_env(render_mode: str = None):
        env = gym.make(env_id, config=env_cfg, render_mode=render_mode)
        env = WorldFrameObservations(env)
        env = SeedWrapper(env=env, seed=seed)
        return env

    env = make_env()
    print("seed:", seed)

    # abstract environment
    abstraction = create_env_abstraction(env=env)
    system = lambda: System(
        id=env_id,
        variables=abstraction.variables,
        controls=abstraction.controls,
        dynamics=abstraction.dynamics,
        domains=abstraction.domains,
    )

    # create data generator from abstracted system
    data_gen = create_data_generator(system=system())

    # learn valid cbf
    config = CegisConfig(
        EXP_NAME=exp_name,
        SYSTEM=system,
        DOMAINS=abstraction.domains,
        DATA_GEN=data_gen,
        CERTIFICATE=CertificateType.CBF,
        VERIFIER=VerifierType.DREAL,
        VERIFIER_TIMEOUT=verifier_timeout_s,
        VERIFIER_N_CEX=verifier_n_cex,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=max_cegis_iters,
        N_DATA=n_data_samples,
        SEED=seed,
        LOGGER=LoggerType.AIM,
        N_EPOCHS=n_cegis_epochs,
        LEARNING_RATE=learning_rate,
        WEIGHT_DECAY=weight_decay,
        LOSS_RELU=loss_relu
    )
    cegis = Cegis(config=config, verbose=verbose)
    result = cegis.solve()

    # reinforcement learning with learned cbf
    rl_args = ppo_args
    rl_args.env_id = make_env
    run(rl_args)


    print(system().id)
    print(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
