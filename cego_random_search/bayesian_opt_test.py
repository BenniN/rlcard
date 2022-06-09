import os
import torch
import json

import rlcard
from rlcard.agents import DQNAgent
from rlcard.agents.random_agent import RandomAgent

from rlcard.games.cego.utils import save_args_params

from rlcard.utils import (
    tournament,
    reorganize,
    Logger,
    plot_curve,
    get_device,
    set_seed,
)  # import some useful functions

from ray import tune  # tuning framework
from ray.tune.suggest.hebo import HEBOSearch
from ray.tune.schedulers import AsyncHyperBandScheduler

config = {
    'env_name': 'cego',
    'game_variant': 'standard',
    'game_activate_heuristic': True,
    'game_judge_by_points': 1,
    'seed': 12,
    'num_episodes': 50000,
    'num_eval_games': 1000,
    'evaluate_every': 500,
    'log_dir': 'bayesian_opt_test',
    'replay_memory_size': tune.qlograndint(10000, 200000, 10000),
    'update_target_estimator_every': tune.qlograndint(1000, 20000, 1000),
    'discount_factor': tune.uniform(0.5, 0.99),
    'epsilon_start': 1.0,
    'epsilon_end': tune.uniform(0.001, 0.1),
    'epsilon_decay_steps': 20000,
    'batch_size': tune.choice([32, 64, 128]),
    'layer_1': 512,
    'layer_2': tune.choice([512, 256, 128]),
    'layer_3': tune.choice([512, 256, 128, None]),
    'learning_rate': tune.uniform(1e-6, 1e-2),
}

max_concurrency = 8

def train_dqn(config):
    device = get_device()
    print(device)

    set_seed(config['seed'])

    # Make the environment with seed
    env = rlcard.make(
        config['env_name'],
        config={
            'seed': config['seed'],
            'game_variant': config['game_variant'],
            'game_activate_heuristic': config['game_activate_heuristic'],
            'game_judge_by_points': config['game_judge_by_points']
        }
    )

    # # this is our DQN agent
    dqn_agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[config['layer_1'], config['layer_2'], config['layer_3']],
        device=device,
        replay_memory_size=config['replay_memory_size'],
        update_target_estimator_every=config['update_target_estimator_every'],
        discount_factor=config['discount_factor'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay_steps=config['epsilon_decay_steps'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )

    agents = [dqn_agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))

    env.set_agents(agents)  # set agents to the environment

    save_args_params(config)

    # Start training
    with Logger(config['log_dir'] + '/model') as logger:
        for episode in range(config['num_episodes']):

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                dqn_agent.feed(ts)

            # Evaluate the performance.

            if episode % config['evaluate_every'] == 0:
                reward = tournament(
                    env,
                    config['num_eval_games'],
                )[0]

                logger.log_performance(
                    env.timestep,
                    reward
                )

                tune.report(reward=reward)

        csv_path, fig_path = logger.csv_path, logger.fig_path

    plot_curve(csv_path, fig_path, "DQN")
    # Save model
    save_path = os.path.join(
        config['log_dir'] + '/model_', 'model.pth')
    torch.save(dqn_agent, save_path)
    print('Model saved in', save_path)

    config['index'] += 1


if __name__ == '__main__':
    isExist = os.path.exists(config['log_dir'])
    if not isExist:
        os.makedirs(config['log_dir'])


    algo = HEBOSearch(
        random_state_seed=config['seed'],
        max_concurrent=max_concurrency
    )

    scheduler = AsyncHyperBandScheduler()

    analysis = tune.run(
        train_dqn,
        scheduler=scheduler,
        config=config,
        search_alg=algo,
        metric="reward",
        mode="max",
        num_samples=50,
        time_budget_s=21600
    )

    dfs = analysis.trial_dataframes

    best_trial = analysis.best_trial  # Get best trial
    best_config = analysis.best_config # Get best config
    best_result_df = analysis.best_result_df # Get best result dataframe
    print("Best trial config: {}".format(best_trial))
    print("Best trial final reward: {}".format(best_config))
    print("Best trial result dataframe: {}".format(best_result_df))

    # with open(config['log_dir']+'/result_best_config.json', 'w') as f:
    #     json.dump(best_trial.config, f, indent=4)

    # with open(config['log_dir']+'/result_best_reward.json', 'w') as f:
    #     json.dump(best_trial.last_result["reward"], f, indent=4)

