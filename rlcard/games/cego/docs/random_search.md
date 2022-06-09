# Random Grid Search

## First Round

Iterations of Random Argumentsets: 20

### Parametersets:

- env_name: "cego"
- game_judge_by_points: 1,
- game_variant: "standard"
- game_activate_heuristic: True
- seed: 12
- replay_memory_size: 50000, 100000, 200000
- update_target_estimator_every: 1000, 2000, 10000
- discount_factor: 0.75, 0.8, 0.95, 0.99
- epsilon_start: 1.0
- epsilon_end: 0.1, 0.05, 0.01
- epsilon_decay_steps: 100000
- batch_size: 32, 64
- mlp_layers: [512, 512, 512], [512, 256, 128], [512, 512]
- num_eval_games": [1000],
- num_episodes": [100000],
- evaluate_every": [1000],
- learning_rate: [0.0001, 0.00005, 0.00001, 0.000005]

**After that a tournament was made with**:

- Each Resulting Model against random agents 5 times.
- 1000 Games per tournament
- 5 diverent seeds: 12, 17, 20, 30, 33
- average Reward

Model Ranking:

1. model 2: 0.512
2. model 16: 0.511
3. model 7: 0.506
4. model 4: 0.5044
5. model 14: 0.504

## 2. Round

Take 5 best models and combine their parameters. 10 Iterations of grid search.

Parametersets:

- env_name: "cego"
- game_judge_by_points: 1,
- game_variant: "standard"
- game_activate_heuristic: True
- seed: 12
- replay_memory_size: 50000, 100000, 200000
- update_target_estimator_every: 2000, 10000
- discount_factor: 0.75, 0.99
- epsilon_start: 1.0
- epsilon_end: 0.1, 0.01, 0.05
- epsilon_decay_steps: 100000
- batch_size: 32, 64
- mlp_layers: [512, 512, 512], [512, 512], [512, 256, 128]
- num_eval_games": [1000],
- num_episodes": [100000],
- evaluate_every": [1000],
- learning_rate: 0.0001, 0.00001, 0.00005

## Further Testing after the first tests. Another game pointing variant was tested for 10 episodes

- game_judge_by_points: 0 - Count with cego points
- game_judge_by_points: 2 - Count won games