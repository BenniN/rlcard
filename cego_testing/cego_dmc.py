import torch

import rlcard
from rlcard.agents.dmc_agent import DMCTrainer

from rlcard.games.cego.utils import save_args_params

from rlcard.utils import (
    set_seed,
    get_device
)  # import some useful functions

args = {
    'env_name': 'cego',
    'game_variant': 'standard',
    'game_judge_by_points': 2,
    'game_activate_heuristic': True,
    'cuda': '',
    'seed': 12,
    'load_model': True,
    'xpid': 'dmc',
    'save_interval': 30,
    'num_actor_devices': 1,
    'num_actors': 6,
    'training_device': '0',
    'log_dir': 'experiments/cego_dmc_standard',
    'total_frames': 100000000000,
    'exp_epsilon': 0.01,
    'batch_size': 32,
    'unroll_length': 100,
    'num_buffers': 50,
    'num_threads': 4,
    'max_grad_norm': 40,
    'learning_rate': 0.0001,
    'alpha': 0.99,
    'momentum': 0,
    'epsilon': 0.00001
}


def train(env_name, game_variant, game_judge_by_points, game_activate_heuristic,
          cuda, seed, load_model, xpid, save_interval, num_actor_devices, num_actors,
          training_device, log_dir, total_frames, exp_epsilon,
          batch_size, unroll_length, num_buffers, num_threads,
          max_grad_norm, learning_rate, alpha, momentum, epsilon):

    device = get_device()
    print(device)

    set_seed(seed)

    env = rlcard.make(
        env_name,
        config={
            'seed': seed,
            'game_variant': game_variant,
            'game_activate_heuristic': game_activate_heuristic,
            'game_judge_by_points': game_judge_by_points
        }
    )

    trainer = DMCTrainer(
        env=env,
        cuda=cuda,
        xpid=xpid,
        load_model=load_model,
        save_interval= save_interval,
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
    save_args_params(args)
    train(**args)
