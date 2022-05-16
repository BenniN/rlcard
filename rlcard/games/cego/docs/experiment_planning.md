# Experiment Planning

## Available Algorithms

- DQN (Deep Q-Network)
- NFSP (Neural Fictitious Self-Play
- DMC (Deep Monte-Carlo)

## Parameters Tuning

[@zha\_rlcard\_2020](http://arxiv.org/abs/1910.04376), [@geron\_hands-machine\_2017](https://scholar.google.com/scholar?q=Hands-On%20Machine%20Learning%20with%20Scikit-Learn%20and%20TensorFlow%3A%20Concepts%2C%20Tools%2C%20and%20Techniques%20for%20Building%20Intelligent%20Systems%2C%20G%C3%A9ron), [@sutton\_reinforcement\_2018](https://scholar.google.com/scholar?q=Reinforcement%20Learning%2C%20second%20edition%3A%20An%20Introduction%2C%20Sutton), [@mnih\_playing\_2013](http://arxiv.org/abs/1312.5602), [@heinrich\_deep\_2016](http://arxiv.org/abs/1603.01121)

### DQN

- **Replay Memory Size** (int): Size of the experience replay memory (used for decoupling consecutive samples)
- **update\_target\_estimator_every**: Step-Interval in which Q-Estimator is updated to target estimator
- **discount factor** (float): Rate at which future rewards matter within reward calculation vor Q-Values
- **epsilon_start** (float): epsilon value at the start (Chance of taking a random action rather then the best action). This value decays over time
- **epsilon_end** (float): Min epsilon at the end of decay
- **epsilon\_decay\_steps** (int): Number of steps to decay epsilon.
- **batch size** (int): Size of batches to sample from replay memory
- **evaluate every** (int): How often to evaluate the network (Doesnt have any effect on training, only on documentation).
- **train every** (int): Training interval (most likely the value of 1).
- **mlp layer** (list): Size of Multilayer Perceptron
- **Learning Rate** (float): Rate at which the modell is adjusted
- **device** (torch.device): whether to use the cpu or gpu

### NFSP

- **... DQN Parameters**
- **hidden layers sizes** (list): The Hidden Layer Size of the sl-network/ average-policy Network
- **reservoir buffer capacity** (int): The size of the buffer for average policy
- **anticipatory param** (float): Param to balance sl and rl policy.
- **batch size** (int): The batch size for training average-policy.
- **train every**: Train SL-Policy Interval
- **rl learning rate**: Learning Rate of the RL-Agent (RL-Agent = DQN)
- **sl learning rate**: Learning Rate of the Average Policy
- **min\_buffer\_size\_to\_learn**: The Minimum Buffer size to learn for average policy.

Within NFSP there also lies the problem, that the original Agent consists of 2 NN. One for the average policy and one for the RL-Policy [@heinrich_deep_2016](http://arxiv.org/abs/1603.01121). The original paper also recommends to train with one NFSP-Agent per player. That would mean that for one training 8 NN would be involved. 

Discussion.

### DMC

- **num actor devices** (int): The number devices used for simulation
- **num actors** (int): Number of actors for each simulation device
- **total frames** (int): Total environment frames to train for
- **exp epsilon** (float): The prbability for exploration
- **batch size** (int): Learner batch size
- **unroll length** (int): The unroll length (time dimension)
- **num buffers** (int): Number of shared-memory buffers
- **num threads** (int): Number learner threads
- **max grad norm** (int): Max norm of gradients
- **learningrate** (float): Learning rate
- **alpha** (float): RMSProp smoothing constant
- **momentum** (float): RMSProp momentum
- **epsilon** (float): RMSProp epsilon

DMC:
* An episode is randomly sampled
* Calculate Q Values
* Update Q Table with return averages
* parallelizable

## Adjustability within the game environment

### State Management

Another important factor might be, how the state is represented within the network or in other words, what the network sees from the environment.
The current representation of the so called observation state is (6, 54) Tuple. Each of the 6 planes represents a different aspect of the game. Each plane has the size 54. Every Index represents a different card within the game.

### A Description of each plane:

| indexes | Description |
| --- | --- |
| 0-53 | The cards on the players hand |
| 54-107 | The card that currently wins the trick |
| 108-161 | All the cards within the current trick |
| 162-215 | All the cards that haven't been played jet and may still be played by other players |
| 216-227 | **This plane encodes other game specific information.**<br>**\[216-219\]**: The players within the same team have the value 1.<br>**\[220-223\]**: The player who would win the round is encoded.<br>**\[224-227\]**: The player who started the current trick round is encoded. |

### Alternatives for Encoding:

#### Alternative 1

It may make sense to encode the information about the current trick differently. One possibility would be to encode each trick card in a different plane.

1.  Trick Plane: First Trick Card
2.  Trick Plane: Second Trick Card
3.  Trick Plane: Third Trick Card
4.  Trick Plane: Winning Trick Card

In this case the temporal information of the trick is still kept. For the last card of the trick obviously there is no encoding needed, because until that state the trick is already played and a new trick has begone. The 4. Plane may also be skippable.

#### Alternative 2

Plane 4 can be inverted so it becomes all the cards, that are out of the game.

#### Alternative 3

The target card may be added.

#### Alternative 4

- Base: **Alternative 1**.
- Remove the winning card from encoding.

### Adjustability within payoff

The calculation of the game payoffs may have an effect on training. 3 Variants are feasible.

1.  Use the points each player gets
    - Maximum of points is 79.
    - A player has won, when he has 40 points.
    - Each player of the same team gets the same amount of points.
2.  A win get the player 1 point. A lose gets the player 0 points.
    - Each player of the same team gets the same amount of points.
3.  A win get the player 1 point. A lose gets the player -1 points.
    - Each player of the same team gets the same amount of points.

Variant 1 may lead towards the player taking risky moves to achieve higher overall moves. The Other 2 Variants may lead towards saver play. The last 2 Variants are very similar, with the difference that the 3 Variants is easier to analyse within learning curves, because the break even point at, which party wins more is **0**, meanwhile the break even point for Variant 2 is **0.5**.

## Testing the environment

At First simple experiments where taken to analyse how the Environment acts.

### Fairness within game

Because Cego is a game where typically 3 players play against 1 player, the question may occur, wheather any party has an advantage within the game. This was analysed within a simple experiment.

#### The Experiment

- Random Agents play against each other in 1,000,000 Evaluation games.
- The Average Reward is analysed.
- Game Variant: "Cego"

#### The Result

- The Cego Player received an average Payoff of **-0.07743**
- The other players received an average Payoff of **0.07743**
- This Experiment was repeated 3 times, with similar results.
- The 2. and 3. time the result was **-/+0.076712** and **-/+0.07748**

This heavily implies, that the other players have a slight advantage against the Cego player even thought the Cego player gets a slight head start in points. This is plausible because the within the trick there are 3 players that have a chance to win that trick against one other player.

An interesting question would if the Cego player has a better chance to improve this margin. Because he always has more knowledge about the cards than the other players. There may be more room for improvement because of that.

These results imply that rather than beating the **0** Reward mark. The goal should be to get more then about **-0.077** of average reward.

The problem with this evaluation is, that it also takes games into account, that are not typically played as a Cego rounds. For this case a simple heuristic was implemented. It is defined that a the Cego player should play a variant of Cego when he has at least **15 points** on his hand. With this premise in mind the previous experiment was repeated.

The Result is as follows:

- The Cego player received an average of **-0.044836** points.
- The other players received an average of **+0.044836** points.

Through repeating this experiment with the implemented heuristic we indeed can see an improvement in points for the Cego player. This shows that the heuristic has the intended effect of improving the chances for the Cego player.

### Training of models for particular players

Another question relevant for the understanding of the environment could be if trained models for one player can be used and replaced for the 3 other players. For this question a model was trained

Notable Parameters:

- **The Model was trained in position of the player 0**
- **Architecture**: DQN
- **learning rate**: 0.0001
- **replay memory size**: 100000
- **discount factor**: 0.95
- **mlp layers**: \[512, 512\]
- **num_episodes**: 400,000

#### Results

**Model as Player 0**

- *Player Party Reward*: **-0.02202**
- *Other Party Reward*: **0.02202**

Despite the reward still being negative, it is still an improvement compared to the *~-0,077* expected from a random player.

**Model as Player 1**

- *Player Party Reward*: **0.10226**
- *Other Party Reward*: **-0.10226**

Again despite that the model has not been trained for this particular player, there is still an improvement visible compared with random players.

This Phenomenon may be fourther analysed.

## Algorithm Evaluation

While **DQN** and **NFSP** may need proper Hyper parameter-Tuning [@zha\_rlcard\_2020](http://arxiv.org/abs/1910.04376). DMC seems to get better over time [@zha\_douzero\_2021](http://arxiv.org/abs/2106.06135).

First For **DQN** and **NFSP** it should be check, what impact the position of the player has. The first player is always the cego player. The other players are the opponents in counterclockwise play direction, relative to player 0, the cego player. The Players with indexes 1-3 are in the same team. 

DMCTrainer class trains for all players.

Idea:

- Train Agent for each index: 0-3
- Take the Agent from index 1 replace him with the positions 2 and 3.
- Take the trained models from 2 and 3 and compare, weather the Agent 1 receives roughly the same rewards.
    - If this is the case. It may be feasible for stage 2 to just train 2 Agents for each Algorithm.
    - If its not the case. Every Algorithm needs 4 Trained Models. One for each player.

### Idea

- 4 Evaluation Stages.
- For all stages one Game Variant (Cego) will be used for training

### How to select the proper hyper parameters

#### Grid Search

- Set of Parameters is defined and all possibilities are tested
- With more possible Parameters the computing time rises exponentially.
- Often based on experience values

#### Random Search

- Select Random Values, within a range, and test them with each other.
- you might select good parameters, but same goes the other way
- Still has the same scaling problem as grid search

#### Sampling Random Combinations

* Create A set of different hyper parameters
* sample random combinations from that set
* less computation then grid search

#### Bayesian Optimization

[@frazier\_tutorial\_2018](http://arxiv.org/abs/1807.02811)

- Set of Machine Learning Method with the goal of optimizing a black box model.
- Uses a system rather then brute force to optimize the hyper parameters.
- Based on Bayes Theorem
- Is useful ...
    - for Environments with many hyper parameters (which is the case for these models).
    - When each models computing effort is large
- Is most likely harder to implement then the other

PS: Still needs more research.

### Stage 1: Hyper Parameter Tuning

1.  This Stage requires an evaluation of what parameters are important and what possible Values
2.  Hyper parameter tune *each Algorithms* most important parameters (don't forget the observation-state)
3.  This Stage requires an evaluation of what parameters are important and what possible Values there are
4.  Training Epochs each: 1e3-1e5
5.  Eval every : 20 Epochs
6.  Eval Games: 1000 (Not to many because evaluation games decelerate training time)
7.  Training against random agents

### Stage 2: Training of final DQN and NFSP Model

1.  Take the best Hyper parameter-Sets of each Algorithm out of **Stage 1** and continue
2.  Train for 1e5 - 5e5 Epochs each.
3.  Eval every : 50 Epochs
4.  Eval Games: 1000
5.  Training is against random agents

### Stage 3: Training of DMC Model

1.  Train a DMC Model for **more then 1 Day**
2.  Use the Hyper parameters from DouZero Paper [@zha\_douzero\_2021](http://arxiv.org/abs/2106.06135) or use hype parameter tuning.

**The result should now be 3 Models: DQN, NFSP, DMC**

### Stage 4: Evaluation

Compare The Models resulting from **Stage 2** and **Stage 3** within the RLCard Tournament mode.

1.  Against random agents
2.  Against each other
3.  Tournament Games each: 1e4-1e6

## Further Ideas

- Other Games can use existing models and AIs for comparing trained models
- There is one existing Ai of cego, which is from the old cego-online version.

### Idea

- Take old game AI, create RLCard - Agent from it and use id to compare against other trained Algorithms

#### Problem

- AI is implemented in ActionScript
- Implementing the AI within RLCard might require to much time for planning and realization.