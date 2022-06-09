import os

import torch
import rlcard
from rlcard.agents import (
    DQNAgent,
    NFSPAgent,
    RandomAgent
)

from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
)

args = {
    "_seed": 12,
    "_models": ["experiments/cego_dmc_standard/dmc/0_56806400.pth",
                "random", 
                "random", 
                "random"],
    "_env_name": "cego",
    "_game_variant": "standard",
    "_game_judge_by_points": 0,
    "_game_activate_heuristic": True,
    "_num_games": 1000,
}


def load_model(model_path, env=None, position=None, device=None):
    agent= None
    if os.path.isfile(model_path):  # Torch model
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)

    return agent


def evaluate(_seed, _models, _env_name, _game_variant, _game_judge_by_points, _game_activate_heuristic, _num_games):

    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(_seed)

    all_rewards= []

    # Make the environment with seed
    env = rlcard.make(
        _env_name,
        config={
            'seed': _seed,
            'game_variant': _game_variant,
            'game_judge_by_points': _game_judge_by_points,
            'game_activate_heuristic': _game_activate_heuristic
        }
    )

    # Load models
    agents = []
    for position, model_path in enumerate(_models):
        agents.append(load_model(model_path, env, position, device))
    env.set_agents(agents)

    # Evaluate
    rewards = tournament(env, _num_games)
    for position, reward in enumerate(rewards):
        print(position, _models[position], reward)


if __name__ == '__main__':
    evaluate(**args)
