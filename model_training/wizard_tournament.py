import os

import torch
import rlcard

from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
)

args = {
    "_seed": None,
    "_models": ["final_models/dqn_5e06_mil_seed42/model_round_20.pth",
                "final_models/dqn_5e07_mil_seed42/model_round_20.pth",
                "random"],
    "_env_name": "wizard",
    "_game_judge_by_points": 0,
    "_num_cards": 60,
    "_num_players": 3,
    "_num_games": 10000,
}


def load_model(model_path, env=None, position=None, device=None):
    agent = None
    if os.path.isfile(model_path):  # Torch model
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)

    return agent


def evaluate(_seed, _models, _env_name, _game_judge_by_points, _num_games, _num_players, _num_cards):

    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(_seed)

    all_rewards= []

    max_rounds = int(_num_cards / _num_players)
    # rounds_to_evaluate = [7]
    # rounds_to_evaluate = [int(max_rounds / 2)]
    rounds_to_evaluate = [max_rounds]
    for eachround in rounds_to_evaluate:

        # Make the environment with seed
        env = rlcard.make(
            _env_name,
            config={
                'seed': _seed,
                'game_judge_by_points': _game_judge_by_points,
                'num_cards': _num_cards,
                'num_players:': _num_players
            }
        )

        # Load models
        agents = []
        for position, model_path in enumerate(_models):
            agents.append(load_model(model_path, env, position, device))
        env.set_agents(agents)

        # Evaluate
        rewards = []
        rewards.append(tournament(env, _num_games))
        for position, reward in enumerate(rewards):
            print(position, _models[position], reward)


if __name__ == '__main__':
    evaluate(**args)
