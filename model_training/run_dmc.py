import torch
import time

import rlcard
from rlcard.agents.dmc_agent import DMCTrainer

from rlcard.games.wizard.utils import save_args_params

from rlcard.utils import (
    get_device,
    set_seed,
)  # import some useful functions

args = {
    'env_name': 'wizard',
    'game_judge_by_points': 0,
    'cuda': '',
    "num_cards": 60,
    "num_players": 3,
    "seed": 42,
    'load_model': True,
    'xpid': 'dmc',
    'save_interval': 1000,
    'num_actor_devices': 5,
    'num_actors': 3,
    'training_device': 'cuda:0',
    'log_dir': 'final_dmc_models/dmc_5e06_round20_06_3p',
    'total_frames': 10000000,
    'exp_epsilon': 0.01,
    'batch_size': 32,
    'unroll_length': 100,
    'num_buffers': 50,
    'num_threads': 6,
    'max_grad_norm': 40,
    'learning_rate': 5e-06,
    'alpha': 0.99,
    'momentum': 0,
    'epsilon': 0.00001,
    'game_anticipate_max_param': 0.6

}


def train(env_name, game_judge_by_points, num_cards, num_players,
          cuda, seed, load_model, xpid, save_interval, num_actor_devices, num_actors,
          training_device, log_dir, total_frames, exp_epsilon,
          batch_size, unroll_length, num_buffers, num_threads,
          max_grad_norm, learning_rate, alpha, momentum, epsilon, game_anticipate_max_param):

    device = get_device()
    print(device)

    set_seed(seed)

    max_rounds = int(num_cards / num_players)
    # rounds_to_evaluate = [7]
    # rounds_to_evaluate = [int(max_rounds / 2)]
    rounds_to_evaluate = [20]
    for eachround in rounds_to_evaluate:
        env = rlcard.make(
            env_name,
            config={
                'seed': seed,
                'game_judge_by_points': game_judge_by_points,
                'game_num_players': num_players,
                'game_num_rounds': eachround,
                'game_anticipate_max_param': game_anticipate_max_param
            }
        )

        print("input_shape:", env.state_shape)

        trainer = DMCTrainer(
            env=env,
            cuda=cuda,
            xpid=xpid,
            load_model=load_model,
            save_interval=save_interval,
            num_actor_devices=num_actor_devices,
            num_actors=num_actors,
            training_device=training_device,
            savedir=log_dir,
            total_frames=total_frames,
            exp_epsilon=exp_epsilon,
            batch_size=batch_size,
            unroll_length=unroll_length,
            num_buffers=num_buffers,
            num_threads=num_threads,
            max_grad_norm=max_grad_norm,
            learning_rate=learning_rate,
            alpha=alpha,
            momentum=momentum,
            epsilon=epsilon
        )

        trainer.start()


if __name__ == '__main__':
    start = time.time()
    save_args_params(args)
    train(**args)
    end = time.time()
    total_time = end - start
    print("Execution Time:", total_time)
