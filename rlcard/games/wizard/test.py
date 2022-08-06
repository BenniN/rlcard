from game import WizardGame
from rlcard import agents
from rlcard.agents import RandomAgent
import rlcard
import card
import utils
import os

from rlcard.utils import (
    get_device,
    tournament
)

from rlcard.agents.human_agents.wizard_human_agent import HumanAgent


def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)

    return agent


if __name__ == '__main__':
    # env = WizardGame()
    # testgame.init_game()
    # for p in testgame.players:
    #     playerhand = utils.cards2list(p.hand)
    #     print(p.player_id)
    #     card.WizardCard.print_cards(playerhand)

    wizardEnv = rlcard.make(
        'wizard',
        config={
            'seed': None,
            'game_num_players': 3,
            'game_num_rounds': 5,
            'game_anticipate_max_param': 0.5
        }
    )

    device = get_device()

    env_agents = []
    env_agents.append(HumanAgent(num_actions=wizardEnv.num_actions))
    for _ in range(1, wizardEnv.num_players):
        env_agents.append(RandomAgent(num_actions=wizardEnv.num_actions))

    wizardEnv.set_agents(env_agents)

    trajectories, payoffs = wizardEnv.run(is_training=False)
    print("Trajectory", trajectories[-1][-1])
    print("Payoffs:", payoffs)

    # rewards = tournament(wizardEnv, 100)
    # print("Rewards:", rewards)
