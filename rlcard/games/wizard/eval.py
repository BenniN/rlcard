import matplotlib.pyplot as plt
import csv
import os
import json
import rlcard
import ntpath
import numpy as np
from scipy import stats
import seaborn as sns
import re

from rlcard.agents.random_agent import RandomAgent
from rlcard.games.wizard.utils import ACTION_SPACE, cards2list

from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
)

ROUND_NUM = 11


def load_model(model_path, env=None, position=None, device=None):
    agent = None
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    print("loaded model: {}".format(model_path))

    return agent


def isfloat(num) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def create_cego_dmc_graph(model_path) -> None:
    '''
    Creates a training graph for a dmc_model
    '''
    file = open(model_path + '/dmc/logs.csv')

    csvreader = csv.DictReader(file)

    y = []
    x_cego = []
    x_other = []
    tick = 0

    for row in csvreader:
        if isfloat(row['mean_episode_return_1']):
            x_cego.append(float(row['mean_episode_return_0']))
            y.append(tick)
            tick += 1
        if isfloat(row['mean_episode_return_1']):
            x_other.append(float(row['mean_episode_return_1']))

    fig, ax = plt.subplots()
    ax.plot(y, x_cego, label='Cego Player')
    ax.plot(y, x_other, label='Other Players')
    ax.set(xlabel='Tick', ylabel='reward')
    ax.legend()
    ax.grid()

    fig.savefig(model_path + '/fig.png')

def create_wizard_dqn_graph(model_path) -> None:
    file = open(model_path + '/performance.csv')
    csvreader = csv.DictReader(file)

    y = []
    x_dqn = []
    x_other = []
    tick = 0

    for row in csvreader:
        if isfloat(row['reward']):
            x_dqn.append(float(row['reward']))
            y.append(tick)
            tick += 1

    fig, ax = plt.subplots()
    ax.plot(y, x_dqn, label='Cego Player')
    ax.plot(y, x_other, label='Other Players')
    ax.set(xlabel='Tick', ylabel='reward')
    ax.legend()
    ax.grid()

    fig.savefig(model_path + '/fig.png')



def create_combined_graph(path_to_models, data_per_graph=10):
    '''
    Combines model graphs of specified path into one graph
    '''
    model_dirs = [x[0] for x in os.walk(path_to_models)]

    ys = []
    xs = []

    i = 0

    fig, ax = plt.subplots()
    for model_dir in model_dirs:
        if not os.path.exists(model_dir + '/performance.csv'):
            continue

        i += 1
        ys.append([])
        xs.append([])

        file = open(model_dir + '/performance.csv')

        csvreader = csv.DictReader(file)
        for row in csvreader:
            ys[(i-1) % data_per_graph].append(int(row['timestep']))
            xs[(i-1) % data_per_graph].append(float(row['reward']))

        if i % data_per_graph == 0:
            fig, ax = plt.subplots()
        ax.set(xlabel='timestep', ylabel='reward')
        for idx in range(len(ys)):
            ax.plot(ys[idx], xs[idx], label="model_" +
                    str(idx + i-5), linewidth=2)
        ax.legend()
        ax.grid()

        if i % data_per_graph == 0:
            fig.savefig(path_to_models + '/fig' +
                        str(i//data_per_graph) + '.png', dpi=200)
            ys = []
            xs = []


def play_tournament_and_update_rewards(rewards, game_Settings, path_to_models, num_games, seed=None) -> None:
    '''
    Params:
        rewards (list): Current rewards,
        game_Settings (dict): Dictionary of the game settings
        path_to_models (list): Paths to Models
        num_games (int): Number of Games
        seed (int): The random seed
    '''

    device = get_device()

    env = rlcard.make(
        game_Settings["env_name"],
        config={
            'game_judge_by_points': game_Settings["game_judge_by_points"],
            'game_train_env': game_Settings['game_train_env']
        }
    )

    if seed != None:
        set_seed(seed)
        env.seed(seed)

    agents = convert_to_agents(path_to_models, env, device)
    env.set_agents(agents)

    tournament_reward = tournament(env, num_games)
    for position, rew in enumerate(tournament_reward):
        print(position, path_to_models[position], rew)

    for i in range(len(rewards)):
        rewards[i] += tournament_reward[i]


def convert_to_agents(path_to_models, env, device):
    agents = []
    for position, model_path in enumerate(path_to_models):
        agent = load_model(model_path, env, position, device)
        agents.append(agent)
    return agents


def compare_models_in_tournament(save_path, games_settings, num_games, path_to_models, seeds=None) -> None:
    all_rewards: list = []

    iterations_rewards = [0 for _ in range(len(path_to_models))]
    num_iterations = len(seeds) if seeds != None else 1

    if seeds == None:
        print("Tournament: ", 0)
        print("--------------------------------")
        play_tournament_and_update_rewards(iterations_rewards, games_settings, path_to_models,
                                           num_games)
    else:
        for i in range(len(seeds)):
            print("Tournament: ", i)
            print("--------------------------------")
            play_tournament_and_update_rewards(iterations_rewards, games_settings, path_to_models,
                                               num_games, seeds[i])

    average_rewards = [
        reward / num_iterations for reward in iterations_rewards]

    for i in range(len(path_to_models)):
        all_rewards.append(
            {
                'model': path_to_models[i],
                'avg_reward': average_rewards[i]
            }
        )

    with open(save_path, 'w') as f:
        json.dump(all_rewards, f, indent=4)


def analyse_first_mover_advantage(path, env, num_games):
    relative_vals_over_games = [[], [], [], []]
    total_vals_over_games = [0, 0, 0, 0]
    timesteps = []

    for i in range(num_games):
        print("episode:", i)
        timesteps.append(((i+1)*ROUND_NUM))
        _, _, state = env.run(is_training=False)

        for j in range(len(state['winning_player_history'])):
            starter_id = state['start_player_history'][j]
            winner_id = state['winning_player_history'][j]

            relative_winner_id = (winner_id-starter_id) % 4
            total_vals_over_games[relative_winner_id] += 1

        for j in range(4):
            new_avg = total_vals_over_games[j]/((i+1)*ROUND_NUM)
            relative_vals_over_games[j].append(new_avg)

    fig, ax = plt.subplots()
    ax.set(xlabel='games', ylabel='reward')

    for i in range(4):
        ax.plot(
            timesteps, relative_vals_over_games[i], label="player_"+str(i))
    ax.legend()
    ax.grid()
    fig.savefig(path + "/first_players_advantage_test.png")

    result = {}

    for i in range(4):
        result['player_'+str(i)] = (total_vals_over_games[i] /
                                    (num_games*ROUND_NUM))*100

    result = {k: v for k, v in sorted(
        result.items(), key=lambda item: item[1], reverse=True)}

    with open(path + "/first_players_advantage_result_test.json", 'w') as f:
        json.dump(result, f, indent=4)


def analyse_propability_a_card_wins_a_trick(path, env, num_games):
    trick_wins: dict = {}

    for key in ACTION_SPACE:
        trick_wins[key] = {
            'played': 0,
            'won': 0
        }

    for i in range(num_games):
        print("episode:", i)
        trajectories, payoffs, state = env.run(is_training=False)
        for trick in state['played_tricks']:
            for card in trick:
                trick_wins[str(card)]['played'] += 1

        for card in cards2list(state['winning_card_history']):
            trick_wins[card]['won'] += 1

    result: dict = {}

    for card in trick_wins:
        result[card] = trick_wins[card]['won'] / trick_wins[card]['played']

    sorted_by_prob = {k: v for k, v in sorted(
        result.items(), key=lambda item: item[1], reverse=True)}

    for key in sorted_by_prob:
        sorted_by_prob[key] = round((sorted_by_prob[key]*100), 2)

    with open(path, 'w') as f:
        json.dump(sorted_by_prob, f, indent=4)


def analyse_card_trick_win_propabilities(path, env, num_games):
    '''
        if a trick was won who big is the chance that this card was the winner
    '''
    trick_wins: dict = {}

    for key in ACTION_SPACE:
        trick_wins[key] = 0

    num_trick_wins = 0
    for i in range(num_games):
        print("episode:", i)
        _, _, state = env.run(is_training=False)
        for card in cards2list(state['winning_card_history']):
            trick_wins[card] += 1
            num_trick_wins += 1

    for entry in trick_wins:
        trick_wins[entry] /= (num_games*ROUND_NUM)

    sorted_by_prob = {k: v for k, v in sorted(
        trick_wins.items(), key=lambda item: item[1], reverse=True)}

    for key in sorted_by_prob:
        sorted_by_prob[key] = round((sorted_by_prob[key]*100), 2)

    with open(path, 'w') as f:
        json.dump(sorted_by_prob, f, indent=4)


def compare_rand_search_models_in_tournament(path_to_models, filename, seeds, env_params, num_games) -> None:
    '''
    Params:
        path_to_models (list): paths to models
        filename (str): Name of the result-file
        seeds (list): random Seeds to test for each run
        env_params (dict): Params of Environment,
        num_games (int): Number of games to play for each random seed
    env_params:
        env_name (str): Environment Name
        game_variant (str): Name of the Sub game
        game_judge_by_points (int): Way to judge the points by
        game_activate_heuristic (bool): Use Heuristic for game environments
    '''
    import torch

    model_dirs = [x[0] for x in os.walk(path_to_models)]

    i = 0

    all_rewards = []

    for model_dir in model_dirs:

        if not os.path.exists(model_dir + '/model.pth'):
            continue

        device = torch.device("cpu")

        iterations_rewards = []

        for seed in seeds:
            # Seed numpy, torch, random
            set_seed(seed)

            models = [model_dir + '/model.pth', 'random', 'random', 'random']

            env = rlcard.make(
                env_params['env_name'],
                config={
                    'seed': seed,
                    'game_variant': env_params['game_variant'],
                    'game_judge_by_points': env_params['game_judge_by_points'],
                    'game_activate_heuristic': env_params['game_activate_heuristic']
                }
            )

            agents = []
            for position, model_path in enumerate(models):
                agents.append(load_model(model_path, env, position, device))
            env.set_agents(agents)

            rewards = tournament(env, num_games)
            for position, reward in enumerate(rewards):
                print(position, models[position], reward)

            iterations_rewards.append(
                rewards[0]
            )

        average_rewards = sum(iterations_rewards) / len(iterations_rewards)

        all_rewards.append(
            {
                'model': model_dir,
                'avg_reward': average_rewards
            }
        )

    sort_by_key_and_save_array(
        all_rewards, 'avg_reward', path_to_models+'/' + filename, True)


def sort_by_key_and_save_array(array, key, path, descending=True):
    array.sort(key=lambda x: x[key], reverse=descending)

    with open(path, 'w') as f:
        json.dump(array, f, indent=4)


def get_total_ranking(save_folder, paths_to_models, filename="total_ranking.json") -> None:
    array_rewards: list[dict] = []
    array_slopes: list[dict] = []

    slope_rankings: dict = {}
    reward_rankings: dict = {}
    total_ranks: list[dict] = []

    for path in paths_to_models:
        file = open(path + "/tournament_result.json")

        array_rewards.extend(json.load(file))

        otherfile = open(path + "/lin_reg_slope_result_sorted.json")

        array_slopes.extend(json.load(otherfile))

    array_rewards.sort(key=lambda x: x["avg_reward"], reverse=True)
    array_slopes.sort(key=lambda x: x["slope"], reverse=True)

    for idx, reward in enumerate(array_rewards):
        reward_rankings[reward['model']] = {
            'rank': idx+1,
            'avg_reward': reward['avg_reward'],
        }

    for idx, slope in enumerate(array_slopes):
        slope_rankings[slope['model']] = {
            'rank': idx+1,
            'slope': slope['slope'],
        }

    print(reward_rankings)

    for model in reward_rankings:
        total_ranks.append({
            "model": model,
            "rank": (reward_rankings[model]['rank']*0.5)+(slope_rankings[model]['rank']*0.5),
            "avg_reward": reward_rankings[model]['avg_reward'],
            "slope": slope_rankings[model]['slope']*100000,
        })

    sort_by_key_and_save_array(
        total_ranks, 'rank', save_folder+"/"+filename, False)


def read_performance(model_dir, max=79, min=0) -> tuple[object, object, object, object, object]:
    if not os.path.exists(model_dir + '/performance.csv'):
        return None, None, None, None, None

    file = open(model_dir + '/performance.csv')
    csvreader = csv.reader(file)
    header = next(csvreader)
    x = []
    y = []
    for row in csvreader:
        if(len(row) == 0):
            continue
        x.append(int(row[0]))
        y.append(float(row[1])/float(max+min))

    result = stats.linregress(x, y)
    std = np.std(y)

    return result.slope, result.intercept, std, y, x


def get_ys(x, slope, intercept):
    ys = []
    for x_i in x:
        ys.append(slope*x_i+intercept)

    return ys


def path_leaf(path):
    return ntpath.split(path)


def compare_training_slope(path_to_models, max_value=79, min_value=0):
    model_dirs = [x[0] for x in os.walk(path_to_models)]

    slopes = []
    stds = []

    if not os.path.exists(path_to_models + '/lin_reg_graphs/'):
        os.mkdir(path_to_models + '/lin_reg_graphs/')

    for model_dir in model_dirs:
        slope, intercept, std, y, x = read_performance(
            model_dir, max_value, min_value)
        if x is None:
            continue

        dir_name = path_leaf(model_dir)[1]

        slopes.append({
            'model': model_dir,
            'slope': slope,
        })

        stds.append({
            'model': model_dir,
            'std': std,
        })

        fig, ax = plt.subplots()
        ax.set(xlabel='timestep', ylabel='reward_normalized (max:1, min:0)')
        ax.plot(x, y, label=dir_name, linewidth=2)
        ax.plot(x, get_ys(x, slope, intercept),
                label="linear regression: "+dir_name, linewidth=2)
        ax.legend()
        ax.grid()
        fig.savefig(path_to_models + '/lin_reg_graphs/lin_reg_' +
                    dir_name + '.png', dpi=200)

    sort_by_key_and_save_array(
        stds, 'std', path_to_models+'/std_result_sorted.json', False)

    sort_by_key_and_save_array(
        slopes, 'slope', path_to_models+'/lin_reg_slope_result_sorted.json', True)


def sort_by_key_and_save_array(array, key, path, descending=True):
    array.sort(key=lambda x: x[key], reverse=descending)

    with open(path, 'w') as f:
        json.dump(array, f, indent=4)


def analyse_card_round_position(env, path, num_games, player_id):
    heatmap: dict = {}
    for key in ACTION_SPACE:
        heatmap[key] = [0] * ROUND_NUM

    for i in range(num_games):
        print("episode:", i)
        trajectories, _, _ = env.run(is_training=False)
        actions = [entry[1] for entry in trajectories[0]
                   [-1]['action_record'] if entry[0] == player_id]

        for i in range(len(actions)):
            heatmap[actions[i]][i] += 1

    sns.heatmap(list(heatmap.values()), fmt="d")
    plt.savefig(path)

    # print(ax)

# sorts the files in a natural way


def sorted_alphanumeric(data):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def compare_dmc_checkpoints(env_params, path_to_dmc_models, player_index, num_games, seed):
    files = [f for f in sorted_alphanumeric(os.listdir(
        path_to_dmc_models)) if os.path.isfile(os.path.join(path_to_dmc_models, f))]

    reward_results = []
    checkpoint_files = []

    device = get_device()

    env = rlcard.make(
        env_params['env_name'],
        config={
            'seed': seed,
            'game_variant': env_params['game_variant'],
            'game_judge_by_points': env_params['game_judge_by_points'],
            'game_activate_heuristic': env_params['game_activate_heuristic']
        }
    )

    idx = 0
    for file in files:
        # dir_name = path_leaf(model_dir)
        if(file.startswith(str(player_index)+"_")):
            print(file)
            idx += 1
            checkpoint_files.append(idx)

            agents = []
            dl_agent = None

            for i in range(env.num_players):
                if i == player_index:
                    dl_agent = load_model(
                        path_to_dmc_models+file, env, player_index, device)
                    agents.append(dl_agent)
                else:
                    agents.append(RandomAgent(num_actions=env.num_actions))

            env.set_agents(agents)

            reward_results.append(tournament(env, num_games)[player_index])

    fig, ax = plt.subplots()
    ax.set(xlabel='model', ylabel='avg reward')
    ax.plot(checkpoint_files, reward_results,
            label='average rewards over checkpoints', linewidth=2)
    ax.legend()
    ax.grid()
    fig.savefig(path_to_dmc_models + '/checkpoint_graph_player_' +
                str(player_index) + '.png')