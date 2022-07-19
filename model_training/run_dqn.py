import os
import time

import torch

import rlcard
from rlcard.agents import DQNAgent, NFSPAgent
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

# arguments for the random search
# dqn_point_var_0/model_4

args = {
    # "algorithm": "nfsp",
    # "log_dir": "final_models/nfsp_wizard_player_1",
    "algorithm": "dqn",
    "log_dir": "final_models/dqn_wizard_max_round_four_players",
    "env_name": "wizard",
    "game_judge_by_points": 0,
    "num_cards": 60,
    "num_players": 4,
    "seed": 24,
    "replay_memory_size": 10000,
    "update_target_estimator_every": 1000,
    "discount_factor": 0.95,
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay_steps": 100000,
    "batch_size": 32,
    "mlp_layers": [512, 512],
    "num_eval_games": 1000,
    "num_episodes": 50000,
    "evaluate_every": 100,
    "learning_rate": 0.0001,  # 1*10^-5 oder 0.00001
}


def train(algorithm, log_dir, env_name, num_cards, num_players, game_judge_by_points, seed, replay_memory_size,
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

        if algorithm == 'nfsp':
            pass
            agent = NFSPAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                hidden_layers_sizes=[64, 64],
                q_mlp_layers=[64, 64],
                device=device,
            )
            # # this is our DQN agent
        else:
            agent = DQNAgent(
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
        agents = [agent]
        for _ in range(1, env.num_players):
            agents.append(RandomAgent(num_actions=env.num_actions))

        env.set_agents(agents)  # set agents to the environment

        checkpoint_count = 0

        # Start training
        with Logger(log_dir) as logger:
            for episode in range(num_episodes):

                if algorithm == 'nfsp':
                    agents[0].sample_episode_policy()

                # Generate data from the environment
                trajectories, payoffs = env.run(is_training=True)

                # Reorganaize the data to be state, action, reward, next_state, done
                trajectories = reorganize(trajectories, payoffs)

                # Feed transitions into agent memory, and train the agent
                # Here, we assume that DQN always plays the first position
                # and the other players play randomly (if any)
                for ts in trajectories[0]:
                    agent.feed(ts)

                # Evaluate the performance. Play with random agents.
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
        plot_curve(csv_path, log_dir + "/fig_model_round_" +
                   str(eachround), algorithm)

        # Save model
        save_path = os.path.join(log_dir, 'model_round_' + str(eachround) + '.pth')
        torch.save(agent, save_path)
        print('Model saved in', save_path)


if __name__ == '__main__':
    start = time.time()
    environment = os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(environment)
    train(**args)
    end = time.time()
    total_time = end - start
    print("Execution Time:", total_time)
