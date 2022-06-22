from game import WizardGame
import card
import utils

if __name__ == '__main__':
    testgame = WizardGame()
    testgame.init_game()
    for p in testgame.players:
        playerhand = utils.cards2list(p.hand)
        print(p.player_id)
        card.WizardCard.print_cards(playerhand)