import unittest

import rlcard
from rlcard.games.wizard import Game
from rlcard.games.wizard.utils import cards2value, encode_observation_var1


def check_if_all_cards_are_unique(players: list) -> bool:
    ''' Check if all players have unique cards

    Parameters:
        - players (list): list of players
    '''

    all_cards_set = set()
    count = 0

    count += len(players[0].valued_cards)
    for card in players[0].valued_cards:
        all_cards_set.add(str(card))

    for player in players:
        count += len(player.hand)
        for card in player.hand:
            all_cards_set.add(str(card))

    print("Card count1: ", len(all_cards_set))
    print("Card count2: ", count)

    if len(all_cards_set) == count:
        return True
    else:
        return False


class TestCardUniqueness(unittest.TestCase):
    def test_card_uniqueness(self):
        game = Game()
        game.init_game()

        self.assertTrue(check_if_all_cards_are_unique(game.players))


class TestStartPayoffs(unittest.TestCase):
    def test_start_payoffs(self):
        game = Game()
        game.init_game()

        expected_payoffs = [
            cards2value(game.players[0].valued_cards), 0, 0, 0
        ]

        self.assertEqual(game.points, expected_payoffs)

if __name__ == '__main__':
    unittest.main()
