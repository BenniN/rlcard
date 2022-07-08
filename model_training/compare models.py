import json
import os
import matplotlib.pyplot as plt
from rlcard.games.wizard.utils import load_model
from rlcard.agents import (
    DQNAgent,
    RandomAgent,
    NFSPAgent
)

import rlcard

from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
)

seeds = [12, 17, 20, 30, 33]
env_name = 'wizard'
game_variant = 'wizardGame'
game_judge_by_points = 0
game_activate_heuristic = False
game_train_env = [False, False, False, False]
num_games = 1000

dqn_model_path = "final_models/dqn_wizard_player_0/checkpoint_4/model.pth"
nfsp_model_path = "final_models/nfsp_wizard_player_0/checkpoint_4/model.pth"
dmc_model_path = "final_models/dmc_models/wizard_ultimo/dmc/0_537203200.pth"


def compare_model_in_tournament(path, path_to_models):
    device = get_device()

    all_rewards = []

    iterations_rewards = [0 for _ in range(len(path_to_models))]
    num_iterations = len(seeds)

    for seed in seeds:
        # Seed numpy, torch, random
        set_seed(seed)

        env = rlcard.make(
            env_name,
            config={
                'seed': seed,
                'game_variant': game_variant,
                'game_judge_by_points': game_judge_by_points,
                'game_activate_heuristic': game_activate_heuristic,

            }
        )

        agents = []
        for position, model_path in enumerate(path_to_models):
            agent = load_model(model_path, env, position, device)
            agents.append(agent)
        env.set_agents(agents)

        rewards = tournament(env, num_games)
        for position, reward in enumerate(rewards):
            print(position, path_to_models[position], reward)

        for i in range(len(rewards)):
            iterations_rewards[i] += rewards[i]

    # average_rewards = sum(iterations_rewards) / len(iterations_rewards)

    average_rewards = [
        reward / num_iterations for reward in iterations_rewards]

    for i in range(len(path_to_models)):
        all_rewards.append(
            {
                'model': path_to_models[i],
                'avg_reward': average_rewards[i]
            }
        )

    with open(path, 'w') as f:
        json.dump(all_rewards, f, indent=4)


if __name__ == '__main__':
    # models_dqn = [
    #     dqn_model_path,
    #     "random",
    #     "random",
    #     "random",
    # ]
    # compare_model_in_tournament("final_models/dqn_t_result.json", models_dqn)
    # models_nfsp = [
    #     nfsp_model_path,
    #     "random",
    #     "random",
    #     "random",
    # ]
    # compare_model_in_tournament("final_models/nfsp_t_result.json", models_nfsp)
    models = [
        "random"
    ]
    compare_model_in_tournament(
        "final_models/dmc_t_result_heuristic_utlimo.json", models)
    # models_dqn_vs_nfsp = [
    #     dqn_model_path,
    #     nfsp_model_path,
    #     "random",
    #     "random",
    # ]
    # compare_model_in_tournament(
    #     "final_models/dqn_vs_nfsp_t_result.json", models_dqn_vs_nfsp)
    # models_nfsp_vs_dqn = [
    #     nfsp_model_path,
    #     dqn_model_path,
    #     "random",
    #     "random",
    # ]
    # compare_model_in_tournament(
    #     "final_models/nfsp_vs_dqn_t_result.json", models_nfsp_vs_dqn)

    # models_dmc_vs_dqn = [
    #     dmc_model_path,
    #     dqn_model_path,
    #     "random",
    #     "random",
    # ]
    # compare_model_in_tournament(
    #     "final_models/dmc_vs_dqn_t_result.json", models_dmc_vs_dqn)

    # models_dqn_vs_dmc = [
    #     dqn_model_path,
    #     dmc_model2_path,
    #     "random",
    #     "random",
    # ]
    # compare_model_in_tournament(
    #     "final_models/dqn_vs_dmc_t_result.json", models_dqn_vs_dmc)

    # models_dmc_vs_nfsp = [
    #     dmc_model_path,
    #     nfsp_model_path,
    #     "random",
    #     "random",
    # ]
    # compare_model_in_tournament(
    #     "final_models/dmc_vs_nfsp_t_result.json", models_dmc_vs_nfsp)

    # models_nfsp_vs_dmc = [
    #     nfsp_model_path,
    #     dmc_model2_path,
    #     "random",
    #     "random",
    # ]
    # compare_model_in_tournament(
    #     "final_models/nfsp_vs_dmc_t_result.json", models_nfsp_vs_dmc)
