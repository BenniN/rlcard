import json
import os
import csv
import matplotlib.pyplot as plt

import rlcard

from rlcard.games.cego.utils import load_model

from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
)

path_to_models= 'random_search_results/dqn'

seeds= [12, 17, 20, 30, 33]
env_name= 'cego'
game_variant= 'standard'
game_judge_by_points= 1
game_activate_heuristic= True
num_games= 1000

def create_combined_graph(path_to_models, data_per_graph= 10):
    model_dirs= [x[0] for x in os.walk(path_to_models)]

    ys = []
    xs = []

    i= 0

    fig, ax = plt.subplots()
    for model_dir in model_dirs:
        if not os.path.exists(model_dir+ '/performance.csv'):
            continue

        i+= 1
        ys.append([])
        xs.append([])

        file = open(model_dir+ '/performance.csv')

        csvreader = csv.DictReader(file)
        for row in csvreader:
            ys[(i-1)%data_per_graph].append(int(row['timestep']))
            xs[(i-1)%data_per_graph].append(float(row['reward']))

        if i%data_per_graph== 0:
            fig, ax = plt.subplots()
        ax.set(xlabel='timestep', ylabel='reward')
        for idx in range(len(ys)):
            ax.plot(ys[idx], xs[idx], label= "model_"+str(idx+ i-5), linewidth=2)
        ax.legend()
        ax.grid()

        if i%data_per_graph== 0:
            fig.savefig(path_to_models + '/fig'+ str(i//data_per_graph)+ '.png', dpi=200)
            ys = []
            xs = []

def compare_model_in_tournament(path_to_models):
    model_dirs= [x[0] for x in os.walk(path_to_models)]

    i= 0

    all_rewards= []

    for model_dir in model_dirs:
        
        if not os.path.exists(model_dir+ '/model.pth'):
            continue

        # Check whether gpu is available
        device = get_device()

        iterations_rewards = []

        for seed in seeds:
            # Seed numpy, torch, random
            set_seed(seed)

            models= [model_dir+ '/model.pth', 'random', 'random', 'random']

            env = rlcard.make(
                env_name,
                config={
                    'seed': seed,
                    'game_variant': game_variant,
                    'game_judge_by_points': game_judge_by_points,
                    'game_activate_heuristic': game_activate_heuristic
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

    all_rewards.sort(key= lambda x: x['avg_reward'], reverse= True);

    with open(path_to_models+'/tournament_result.json', 'w') as f:
        json.dump(all_rewards, f, indent=4)

            

if __name__ == '__main__':
    create_combined_graph(path_to_models, 5)
    compare_model_in_tournament(path_to_models)
