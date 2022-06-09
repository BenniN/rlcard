import os

import torch

import rlcard
from rlcard.agents import NFSPAgent
from rlcard.agents.random_agent import RandomAgent

from rlcard.utils import (
    tournament,
    reorganize,
    Logger,
    plot_curve,
    get_device,
    set_seed,
)  # import some useful functions

args = {
    "_log_dir": "experiments/cego_nfsp_result_player/",
    "_env_name": "cego",
    "_game_judge_by_points": 2,
    "_seed": 10,
    "_hidden_layers_sizes": [64, 64],
    "_reservoir_buffer_capacity": 20000,
    "_anticipatory_param": 0.1,
    "_batch_size": 256,
    "_train_every": 1,
    "_rl_learning_rate": 0.1,
    "_sl_learning_rate": 0.005,
    "_min_buffer_size_to_learn":100,
    "_q_replay_memory_size":20000,
    "_q_replay_memory_init_size":100,
    "_q_update_target_estimator_every":1000,
    "_q_discount_factor":0.99,
    "_q_epsilon_start":0.06,
    "_q_epsilon_end":0,
    "_q_epsilon_decay_steps":int(1e6),
    "_q_batch_size":32,
    "_q_train_every":1,
    "_q_mlp_layer": [64, 64],
    "_num_eval_games": 10000,
    "_num_episodes": 1000,
    "_evaluate_every": 100
}


def train(_log_dir, _env_name, _game_judge_by_points, _seed, _hidden_layers_sizes, _reservoir_buffer_capacity,
          _anticipatory_param, _batch_size, _train_every, _rl_learning_rate, _sl_learning_rate,
          _min_buffer_size_to_learn, _q_replay_memory_size, _q_replay_memory_init_size,
          _q_update_target_estimator_every, _q_discount_factor, _q_epsilon_start, _q_epsilon_end,
          _q_epsilon_decay_steps, _q_batch_size, _q_train_every,
          _q_mlp_layer, _num_eval_games=10000, _num_episodes=1000,
          _evaluate_every=100):

    # Check whether gpu is available
    device = get_device()

    set_seed(_seed)

    # Make the environment with seed
    env = rlcard.make(
        _env_name,
        config={
            'seed': _seed,
            'game_judge_by_points': _game_judge_by_points
        }
    )

    agents = []

    nfsp_agent= NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=_hidden_layers_sizes,
            reservoir_buffer_capacity= _reservoir_buffer_capacity,
            anticipatory_param= _anticipatory_param,
            batch_size= _batch_size,
            train_every= _train_every,
            rl_learning_rate= _rl_learning_rate,
            sl_learning_rate= _sl_learning_rate,
            min_buffer_size_to_learn= _min_buffer_size_to_learn,
            q_replay_memory_size= _q_replay_memory_size,
            q_replay_memory_init_size= _q_replay_memory_init_size,
            q_update_target_estimator_every= _q_update_target_estimator_every,
            q_discount_factor= _q_discount_factor,
            q_epsilon_start= _q_epsilon_start,
            q_epsilon_end= _q_epsilon_end,
            q_epsilon_decay_steps= _q_epsilon_decay_steps,
            q_batch_size= _q_batch_size,
            q_train_every= _q_train_every,
            q_mlp_layers=_q_mlp_layer,
            device=device,
        )
    agents.append(nfsp_agent)

    for i in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))

    env.set_agents(agents)

    # Start training
    with Logger(_log_dir) as logger:
        for episode in range(_num_episodes):

            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agents memories, and train the agent
            for ts in trajectories[0]:
                nfsp_agent.feed(ts)

            # Evaluate the performance.s.
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
    plot_curve(csv_path, fig_path, "DQN")

    # Save models
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(nfsp_agent, save_path)
    print('Model saved in', save_path)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    train(**args)
