import os

import torch

import rlcard
from rlcard.agents import NFSPAgent
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

args = {
    "env_name": ["wizard"],
    "game_judge_by_points": [0],
    "num_cards": [60],
    "num_players": [3],
    "seed": [42],
    "hidden_layers_sizes": [[128, 128], [256, 256], [512, 512], [512, 512, 512]],
    "reservoir_buffer_capacity": [20000, 50000, 100000],
    "anticipatory_param": [0.1, 0.25, 0.5],
    "batch_size": [256, 128, 64],
    "train_every": [1],
    "rl_learning_rate": [0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001],
    "sl_learning_rate": [0.005, 0.001, 0.0001, 0.00001],
    "min_buffer_size_to_learn": [100],
    "q_replay_memory_size": [20000, 100000, 200000],
    "q_replay_memory_init_size": [100],
    "q_update_target_estimator_every": [1000, 2000, 10000],
    "q_discount_factor": [0.75, 0.8, 0.95, 0.99],
    "q_epsilon_start": [0.06, 0.1, 1.0],
    "q_epsilon_end": [0, 0.1, 0.01],
    "q_epsilon_decay_steps": [int(50000)],
    "q_batch_size": [32, 64],
    "q_train_every": [1],
    "q_mlp_layer": [[256, 256], [512, 512], [512, 512, 512], [512, 256, 128]],
    "num_eval_games": [1000],
    "num_episodes": [50000],
    "evaluate_every": [500]
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


def train(log_dir, env_name, game_judge_by_points, num_cards, num_players,
          seed, hidden_layers_sizes, reservoir_buffer_capacity,
          anticipatory_param, batch_size, train_every, rl_learning_rate, sl_learning_rate,
          min_buffer_size_to_learn, q_replay_memory_size, q_replay_memory_init_size,
          q_update_target_estimator_every, q_discount_factor, q_epsilon_start, q_epsilon_end,
          q_epsilon_decay_steps, q_batch_size, q_train_every,
          q_mlp_layer, num_eval_games=10000, num_episodes=1000,
          evaluate_every=100):
    # Check whether gpu is available
    device = get_device()

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

        agents = []

        nfsp_agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=hidden_layers_sizes,
            reservoir_buffer_capacity=reservoir_buffer_capacity,
            anticipatory_param=anticipatory_param,
            batch_size=batch_size,
            train_every=train_every,
            rl_learning_rate=rl_learning_rate,
            sl_learning_rate=sl_learning_rate,
            min_buffer_size_to_learn=min_buffer_size_to_learn,
            q_replay_memory_size=q_replay_memory_size,
            q_replay_memory_init_size=q_replay_memory_init_size,
            q_update_target_estimator_every=q_update_target_estimator_every,
            q_discount_factor=q_discount_factor,
            q_epsilon_start=q_epsilon_start,
            q_epsilon_end=q_epsilon_end,
            q_epsilon_decay_steps=q_epsilon_decay_steps,
            q_batch_size=q_batch_size,
            q_train_every=q_train_every,
            q_mlp_layers=q_mlp_layer,
            device=device,
        )
        agents.append(nfsp_agent)

        for _ in range(1, env.num_players):
            agents.append(RandomAgent(num_actions=env.num_actions))

        env.set_agents(agents)

        # Start training
        with Logger(log_dir) as logger:
            for episode in range(num_episodes):

                agents[0].sample_episode_policy()

                # Generate data from the environment
                trajectories, payoffs = env.run(is_training=True)

                # Reorganaize the data to be state, action, reward, next_state, done
                trajectories = reorganize(trajectories, payoffs)

                # Feed transitions into agents memories, and train the agent
                for ts in trajectories[0]:
                    nfsp_agent.feed(ts)

                # Evaluate the performance.s.
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
        plot_curve(csv_path, fig_path, "NFSP_"+ str(eachround))

        # Save models
        save_path = os.path.join(log_dir,  'model' + str(eachround) + '.pth')
        torch.save(nfsp_agent, save_path)
        print('Model saved in', save_path)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "gpu"
    randomSearch(args, 'random_search_results/nfsp_point_var_0', random_search_iterations)
