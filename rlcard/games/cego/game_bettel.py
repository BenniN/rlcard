from rlcard.games.cego import Game

from rlcard.games.cego import Dealer
from rlcard.games.cego import Player
from rlcard.games.cego import JudgerBettel
from rlcard.games.cego import Round
from rlcard.games.cego import Game

from typing import Any


class CegoGameBettel(Game):
    ''' This is the Bettel Variant of Cego

    The Player who plays alone (player 0) is not allowed to win a trick.
    '''

    def init_game(self) -> tuple[dict, Any]:
        self.points = [0 for _ in range(self.num_players)]

        # Initialize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)

        # Initialize players to play the game
        self.players = [Player(i, self.np_random)
                        for i in range(self.num_players)]

        # player 0 is the solo player
        self.players[0].is_single_player = True

        self.judger = JudgerBettel(self.np_random)

        # deal cards to player
        for i in range(self.num_players):
            self.dealer.deal_cards(self.players[i])

        # deal blind cards
        self.blind_cards = self.dealer.deal_blinds()

        # Count the round. There are 4 rounds in each game.
        self.round_counter = 0

        # player who starts the game
        self.current_player = 0

        self.round = Round(self.np_random)
        self.round.start_new_round(0)

        state = self.get_state(self.current_player)

        self.history = []
        self.trick_history = []

        return state, self.round.current_player_idx

    def is_over(self) -> bool:
        if self.points[0] > 0:
            return True
        return self.round_counter >= Game.num_rounds