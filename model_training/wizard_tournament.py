import os

import torch
import rlcard

from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
)

dqn07 ="final_dqn_models/complete_logic/dqn_3players_5e06_07_anticipation_round20/model_round_20.pth"
nfsp07 = "final_nfsp_models/nfsp_5e06_07_3players_round20/model_round_20.pth"
nfsp07_19 ="final_nfsp_models/nfsp_5e06_07_3players_round19/model_round_19.pth"
dmc07 ="final_dmc_models/dmc_wizard_5e06_06_round20_3actor/dmc/1_10054400.pth"
"final_nfsp_models/nfsp_5e06_06_3players_round5/model_round_5.pth"
"final_dqn_models/complete_logic/dqn_5e06_06_anticipation_3players_round5/model_round_5.pth"
"final_dmc_models/dmc_5e06_06_round5/dmc/0_10054400.pth"

dqn07_11 = "final_dqn_models/complete_logic/dqn_5e06_07_anticipation_3players_round19/model_round_19.pth"
args = {
    "_seed": 11,
    "_models": [
                "random",
                 "random",
                 "random",
                "random"],
    "_env_name": "wizard",
    "_num_cards": 60,
    "_num_players": 3,
    "_num_games": 1000,
    "game_anticipate_max_param": 0.7
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


def evaluate(_seed, _models, _env_name, _num_games, _num_players, _num_cards, game_anticipate_max_param):

    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(_seed)

    all_rewards = []

    max_rounds = int(_num_cards / _num_players)
    # rounds_to_evaluate = [7]
    # rounds_to_evaluate = [int(max_rounds / 2)]
    rounds_to_evaluate = [1]
    for eachround in rounds_to_evaluate:

        # Make the environment with seed
        env = rlcard.make(
            _env_name,
            config={
                'seed': _seed,
                'game_num_players': _num_players,
                'game_num_rounds': eachround,
                'game_anticipate_max_param': game_anticipate_max_param
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
