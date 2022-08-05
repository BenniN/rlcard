import os

import torch

import rlcard
from rlcard.agents import DQNAgent
from rlcard.agents.random_agent import RandomAgent

from rlcard.games.wizard.utils import get_random_search_args, args_to_str, save_args_params

from rlcard.utils import (
    tournament,
    reorganize,
    Logger,
    plot_curve,
    get_device,
    set_seed,
)  # import some useful functions

random_search_iterations = 20

# arguments for the random search
args = {
    "env_name": ["wizard"],
    "game_judge_by_points": [0],
    "num_cards": [60],
    "num_players": [3, 4, 5, 6],
    "seed": [42],
    "replay_memory_size": [50000, 100000, 200000],
    "update_target_estimator_every": [1000, 2000, 10000],
    "discount_factor": [0.75, 0.8, 0.95, 0.99],
    "epsilon_start": [1.0],
    "epsilon_end": [0.1, 0.05, 0.01],
    "epsilon_decay_steps": [100000],
    "batch_size": [32, 64],
    "mlp_layers": [[512, 512, 512], [512, 256, 128], [512, 512]],
    "num_eval_games": [1000],
    "num_episodes": [100000],
    "evaluate_every": [1000],
    "learning_rate": [0.0001, 0.00005, 0.00001, 0.000005]
}


def randomSearch(args: dict, random_search_folder: str, random_search_iterations: int):
    set_of_searches = init_search_set(random_search_folder)

    for i in range(len(set_of_searches), random_search_iterations):
        training_args = get_random_search_args(args)
        args_as_string = args_to_str(training_args)

        # rerole as long as args have already been trained
        while args_as_string in set_of_searches:
            training_args = get_random_search_args(args)
            args_as_string = args_to_str(training_args)

        set_of_searches.add(args_as_string)
        training_args["log_dir"] = random_search_folder + "/model_" + str(i)
        save_args_params(training_args)
        train(**training_args)
        save_search_set(random_search_folder, args_as_string)


def train(log_dir, env_name, num_cards, num_players,
          game_judge_by_points, seed, replay_memory_size,
          update_target_estimator_every, discount_factor,
          epsilon_start, epsilon_end, epsilon_decay_steps,
          batch_size, mlp_layers, num_eval_games,
          num_episodes, evaluate_every, learning_rate):
    # Check whether gpu is available
    device = get_device()
    print(device)

    set_seed(seed)

    max_rounds = int(num_cards / num_players)
    # rounds_to_evaluate = [7]
    # rounds_to_evaluate = [int(max_rounds / 2)]
    rounds_to_evaluate = [max_rounds]

    for eachround in rounds_to_evaluate:

        # Make the environment with seed
        env = rlcard.make(
            env_name,
            config={
                'seed': seed,
                'game_judge_by_points': game_judge_by_points,
                'game_num_players': num_players,
                'game_num_rounds': eachround
            }
        )

        # # this is our DQN agent
        dqn_agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=mlp_layers,
            device=device,
            replay_memory_size=replay_memory_size,
            update_target_estimator_every=update_target_estimator_every,
            discount_factor=discount_factor,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            batch_size=batch_size,
            learning_rate=learning_rate

        )
        agents = [dqn_agent]
        for _ in range(1, env.num_players):
            agents.append(RandomAgent(num_actions=env.num_actions))

        env.set_agents(agents)  # set agents to the environment

        # Start training
        with Logger(log_dir) as logger:
            for episode in range(num_episodes):

                # Generate data from the environment
                trajectories, payoffs = env.run(is_training=True)

                # Reorganaize the data to be state, action, reward, next_state, done
                trajectories = reorganize(trajectories, payoffs)

                # Feed transitions into agent memory, and train the agent
                for ts in trajectories[0]:
                    dqn_agent.feed(ts)

                # Evaluate the performance.
                if episode % evaluate_every == 0:
                    logger.log_performance(
                        env.timestep,
                        tournament(
                            env,
                            num_eval_games,
                        )[0]
                    )

            # Get the paths
            csv_path, fig_path = logger.csv_path, logger.fig_path

        # Plot the learning curve
        plot_curve(csv_path, fig_path, "DQN" + str(eachround))

        # Save model
        save_path = os.path.join(log_dir, 'model' + str(eachround) + '.pth')
        torch.save(dqn_agent, save_path)
        print('Model saved in', save_path)


def init_search_set(random_search_folder):
    res = set()
    if os.path.exists(random_search_folder + "/search_set.txt"):
        with open(random_search_folder + "/search_set.txt", "r") as f:
            search_set = set(f.read().splitlines())

            for val in search_set:
                res.add(val)

    return res


def save_search_set(random_search_folder, args_string):
    if not os.path.exists(random_search_folder + "/search_set.txt"):
        open(random_search_folder + "/search_set.txt", 'a').close()

    with open(random_search_folder + "/search_set.txt", 'a') as f:
        f.write(args_string + "\n")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "cpu"
    randomSearch(args, 'random_search_results/dqn_point_var_0', random_search_iterations)
