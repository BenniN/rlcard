# random against human

import rlcard
from rlcard.agents import RandomAgent
from rlcard.agents.human_agents.cego_human_agent import HumanAgent, _print_action

env = rlcard.make(
    'cego',
    config={
        'seed': 12,
        'game_variant': 'standard',
        'game_activate_heuristic': True,
        'game_judge_by_points': 1,
    })

human_agent = HumanAgent(num_actions=env.num_actions)
random1 = RandomAgent(num_actions=env.num_actions)
random2 = RandomAgent(num_actions=env.num_actions)
random3 = RandomAgent(num_actions=env.num_actions)

env.set_agents([
    human_agent,
    random1,
    random2,
    random3,
])

print(env.num_actions)  # 54
print(env.num_players)  # 4
# [[6, 54], [6, 54], [6, 54], [6, 54]]: 4 players, each have the space 6x54
print(env.state_shape)
print(env.action_shape)  # [None, None, None, None]


if __name__ == "__main__":
    trajectories, payoffs = env.run(is_training=False)

    print("Payoffs:", payoffs)

    if payoffs[0] > payoffs[1]:
        print('You win!')
    else:
        print('You lose!')
    print('')
    input("Press any key to continue...")
