from rlcard.games.cego import Game

from rlcard.games.cego import Dealer
from rlcard.games.cego import Player
from rlcard.games.cego import Judger
from rlcard.games.cego import Round

from typing import Any


class CegoGameSolo(Game):
    ''' This is the Solo Variant of Cego

    Other than in Cego, the Solo variant has the following changes:
        - The Single player plays with his hand cards
        - He won't have any knowledge of the blind cards
        - He still will get the points of the blind cards
    '''

    def init_game(self) -> tuple[dict, Any]:
        self.points = [0 for _ in range(self.num_players)]

        # Initialize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)

        # Initialize players to play the game
        self.players = [Player(i, self.np_random)
                        for i in range(self.num_players)]

        # player 0 is not the solo player
        self.players[0].is_single_player = True

        self.judger = Judger(self.np_random)

        # deal cards to player
        for i in range(self.num_players):
            self.dealer.deal_cards(self.players[i])

        # deal blind cards to cego player
        self.blind_cards = self.dealer.deal_blinds()

        # Cego player gets the points from the throw away cards
        self.points = self.judger.receive_points(
            self.points,
            self.players,
            0,
            self.blind_cards  # player get points from blind cards
        )

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
