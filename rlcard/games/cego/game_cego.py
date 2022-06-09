from rlcard.games.cego import Game

from rlcard.games.cego.utils import set_cego_player_deck
from rlcard.games.cego import Dealer
from rlcard.games.cego import Player
from rlcard.games.cego import JudgerStandard
from rlcard.games.cego import Round

from typing import Any


class CegoGameStandard(Game):
    ''' This is the Standard Cego Variant

    - The Cego player will take 2 cards from his hand with him
    - The other cards from his hand are laid aside
    - The Cego player will take the blind cards
    - The Cego player will throw one card away
    This process is currently automated within the function "set_cego_player_deck"
    '''

    def init_game(self) -> tuple[dict, Any]:
        self.points = [0 for _ in range(self.num_players)]

        # Initialize a dealer that can deal cards
        if self.activate_heuristic:
            self.dealer = Dealer(self.np_random, heuristic="cego")
        else:
            self.dealer = Dealer(self.np_random)

        # Initialize players to play the game
        self.players = [Player(i, self.np_random)
                        for i in range(self.num_players)]

        # player 0 is the cego player
        self.players[0].is_single_player = True

        self.judger = JudgerStandard(self.np_random)

        # deal cards to player
        for i in range(self.num_players):
            self.dealer.deal_cards(self.players[i])

        # deal blind cards to cego player
        self.blind_cards = self.dealer.deal_blinds()
        # update cego player deck
        set_cego_player_deck(self.players[0], self.blind_cards)

        # Cego player gets the points from the throw away cards
        self.points = self.judger.update_points(
            self.points,
            self.players,
            0,
            self.players[0].valued_cards
        )
        self.blind_cards= self.players[0].valued_cards

        # Count the round. There are 4 rounds in each game.
        self.round_counter = 0

        # cego player starts the game
        self.current_player = 0

        self.round = Round(self.np_random)
        self.round.start_new_round(0)

        state = self.get_state(self.current_player)

        self.history = []
        self.trick_history = []

        return state, self.round.current_player_idx
