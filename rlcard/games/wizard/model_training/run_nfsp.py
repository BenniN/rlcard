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

# arguments for the random search
# nfsp_poinr_var_0_tuned_dqn/model_12
args = {
    "algorithm": "nfsp",
    "_log_dir": "final_nfsp_playing_Models/Round18to19",
    "_env_name": "wizard",
    "_game_judge_by_points": 0,
    "_num_cards": 60,
    "_num_players": 3,
    "_seed": 42,
    "_hidden_layers_sizes": [512, 512],
    "_reservoir_buffer_capacity": 100000,
    "_anticipatory_param": 0.5,
    "_batch_size": 32,
    "_train_every": 1,
    "_rl_learning_rate": 5e-06,
    "_sl_learning_rate": 0.0001,
    "_min_buffer_size_to_learn": 100,
    "_q_replay_memory_size": 100000,
    "_q_replay_memory_init_size": 100,
    "_q_update_target_estimator_every": 1000,
    "_q_discount_factor": 0.95,
    "_q_epsilon_start": 1.0,
    "_q_epsilon_end": 0.1,
    "_q_epsilon_decay_steps": 100000,
    "_q_batch_size": 32,
    "_q_train_every": 1,
    "_q_mlp_layer": [512, 512],
    "_num_eval_games": 1000,
    "_num_episodes": 100000,
    "_evaluate_every": 1000,
    "game_anticipate_max_param": 0.7
}


def train(algorithm, _log_dir, _env_name, _game_judge_by_points, _seed, _num_cards, _num_players, _hidden_layers_sizes, _reservoir_buffer_capacity,
          _anticipatory_param, _batch_size, _train_every, _rl_learning_rate, _sl_learning_rate,
          _min_buffer_size_to_learn, _q_replay_memory_size, _q_replay_memory_init_size,
          _q_update_target_estimator_every, _q_discount_factor, _q_epsilon_start, _q_epsilon_end,
          _q_epsilon_decay_steps, _q_batch_size, _q_train_every,
          _q_mlp_layer, _num_eval_games, _num_episodes,
          _evaluate_every, game_anticipate_max_param):
    # Check whether gpu is available
    device = get_device()

    set_seed(_seed)

    max_rounds = int(_num_cards / _num_players)
    rounds_to_evaluate = [18,19]

    for _each_round in rounds_to_evaluate:
        # Make the environment with seed
        env = rlcard.make(
            _env_name,
            config={
                'seed': _seed,
                'game_judge_by_points': _game_judge_by_points,
                'game_num_players': _num_players,
                'game_num_rounds': _each_round,
                'game_anticipate_max_param': game_anticipate_max_param
            }
        )

        nfsp_agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=_hidden_layers_sizes,
            reservoir_buffer_capacity=_reservoir_buffer_capacity,
            anticipatory_param=_anticipatory_param,
            batch_size=_batch_size,
            train_every=_train_every,
            rl_learning_rate=_rl_learning_rate,
            sl_learning_rate=_sl_learning_rate,
            min_buffer_size_to_learn=_min_buffer_size_to_learn,
            q_replay_memory_size=_q_replay_memory_size,
            q_replay_memory_init_size=_q_replay_memory_init_size,
            q_update_target_estimator_every=_q_update_target_estimator_every,
            q_discount_factor=_q_discount_factor,
            q_epsilon_start=_q_epsilon_start,
            q_epsilon_end=_q_epsilon_end,
            q_epsilon_decay_steps=_q_epsilon_decay_steps,
            q_batch_size=_q_batch_size,
            q_train_every=_q_train_every,
            q_mlp_layers=_q_mlp_layer,
            device=device,
        )
        agents = [nfsp_agent]
        for _ in range(1, env.num_players):
            agents.append(RandomAgent(num_actions=env.num_actions))

        env.set_agents(agents)  # set agents to the environment

        checkpoint_count = 0

        # Start training
        with Logger(_log_dir) as logger:
            for episode in range(_num_episodes):

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
                    nfsp_agent.feed(ts)

                # Evaluate the performance. Play with random agents.
                if episode % _evaluate_every == 0:
                    logger.log_performance(
                        env.timestep,
                        tournament(
                            env,
                            _num_eval_games,
                        )[0]
                    )

            # Get the paths
            csv_path, fig_path = logger.csv_path, logger.fig_path

        # Plot the learning curve
        plot_curve(csv_path, _log_dir + "/fig_model_round_" +
                   str(_each_round), algorithm)

        # Save model
        save_path = os.path.join(_log_dir, 'model_round_' + str(_each_round) + '.pth')
        torch.save(nfsp_agent, save_path)
        print('Model saved in', save_path)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "cpu"
    train(**args)