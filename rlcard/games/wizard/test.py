from game import WizardGame
from rlcard.agents import RandomAgent
import rlcard
import card
import utils

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
            'seed': 42,
            'game_num_players': 4
        }
    )

    agents = [RandomAgent(num_actions=wizardEnv.num_actions) for _ in range(wizardEnv.num_players)]

    wizardEnv.set_agents(agents)

    wizardEnv.run(is_training=False)
