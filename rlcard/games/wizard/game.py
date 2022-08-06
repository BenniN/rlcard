import random
from copy import deepcopy
import numpy as np
from typing import Any

from abc import ABC
from rlcard.games.wizard.utils import cards2list, get_hand_forecast_value
from rlcard.games.wizard import Dealer
from rlcard.games.wizard import Judger
from rlcard.games.wizard import Round
from rlcard.games.wizard import Player


class WizardGame():
    ''' The class for a wizard game

    class attributes:
        - num_rounds: the number of rounds in a game
        - num_actions: the number of actions a player can play

    instance attributes:
        - allow_step_back (bool): whether to allow step back
        - np_random (np.random.RandomState): numpy random state
        - num_players (int): the number of players
        - points (list[int]): the points of each player
        - dealer (Dealer): the dealer
        - players (list[Player]): the players
        - judger (Judger): the judger
        - round (Round): the current round
        - round_counter (int): the current round counter
        - history (list): the history of the game
        - trick_history (list): the history of the tricks
        - blind_cards (list): the blind cards
        - last_round_winner_idx (int): the last round winner
    '''

    max_num_rounds: int = 15
    num_actions: int = 60

    def __init__(self, allow_step_back=False):

        self.allow_step_back: bool = allow_step_back
        self.np_random: np.random.RandomState = np.random.RandomState()
        self.num_players: int = 6  # there could be 3 to 6 players
        self.points: list[int] = [0 for _ in range(self.num_players)]
        self.max_num_rounds = self.num_actions / self.num_players

        self.dealer: Dealer = None
        self.players: list = None
        self.judger: Judger = None
        self.round: Round = None
        self.round_counter: int = None
        self.history: list = None
        self.trick_history: list = None
        self.trick_winner_card_history: list = None
        self.top_card = None
        self.last_round_winner_idx: int = None
        self.forcast_player_amount: int = None
        self.forcast_predicted_cards: int = None
        self.judge_by_points: int = 0
        self.anticipate_max_param: float = 0.5
        self.trump_color = None

    def configure(self, game_config) -> None:
        ''' Specify some game specific parameters, such as number of players

        game_config = {
            'game_num_players': 4,
            'game_num_rounds': 5,
        }
        '''

        self.num_players = game_config['game_num_players']
        self.max_num_rounds = game_config['game_num_rounds']
        self.points: list[int] = [0 for _ in range(self.num_players)]
        self.with_perfect_information = game_config['game_with_perfect_information']
        self.analysis_mode = game_config['game_analysis_mode']
        self.anticipate_max_param = game_config['game_anticipate_max_param']

    def init_game(self) -> tuple[dict, Any]:
        self.points = [0 for _ in range(self.num_players)]

        # Initialize a dealer that can deal cards

        self.dealer = Dealer(self.np_random)

        # Initialize players to play the game
        self.players = [Player(i, self.np_random)
                        for i in range(self.num_players)]

        self.judger = Judger(self.np_random)

        # Count the round. There are 10 to 20 rounds in each game.
        self.round_counter = 1

        # deal cards to player
        for i in range(self.num_players):
            self.dealer.deal_cards(self.players[i], self.max_num_rounds)

        self.top_card = self.dealer.flip_top_card()

        colors = ['r', 'g', 'b', 'y']

        if self.top_card is None:
            self.top_card = None
        elif self.top_card.suit == "w":
            self.trump_color = colors[random.randint(0, 3)]
        else:
            self.trump_color = self.top_card.suit

        # player starts the game
        self.current_player = self.np_random.randint(0, self.num_players - 1)

        for i in range(self.num_players):
            relative_player_pos = (i - self.current_player) % self.num_players
            self.players[i].forecast = round(
                get_hand_forecast_value(self.anticipate_max_param, self.players[i].hand, self.num_players,
                                        self.max_num_rounds, self.top_card, self.trump_color, relative_player_pos))
            print('forecast_player_' + str(i) + ':', self.players[i].forecast)

        self.round = Round(self.np_random)
        self.round.start_new_round(
            self.current_player, self.num_players, self.top_card, self.trump_color)

        state = self.get_state(self.current_player)

        self.history = []
        self.trick_history = []
        self.trick_winner_card_history = []

        return state, self.round.current_player_idx

    def get_state(self, player_id) -> dict:
        ''' get current state of the game

        Parameters:
            - player_id: the id of the player

        '''

        state = self.round.get_state(self.players[player_id])
        state['num_players'] = self.get_num_players()
        state['current_player'] = self.round.current_player_idx
        state['current_player_forecast'] = self.players[player_id].forecast
        state['current_trick_round'] = self.round_counter
        state['tricks_played'] = self.trick_history
        state['last_round_winner'] = self.last_round_winner_idx
        return state

    def step(self, action) -> tuple[dict, Any]:
        ''' Play a single step in the game

        Parameters:
            - action: the action taken by the current player
        '''

        if self.allow_step_back:
            # save current state for potential step back
            the_round = deepcopy(self.round)
            the_players = deepcopy(self.players)
            the_dealer = deepcopy(self.dealer)
            the_round_counter = deepcopy(self.round_counter)
            the_trick_history = deepcopy(self.trick_history)
            the_trick_winner_card_history = deepcopy(
                self.trick_winner_card_history)
            the_playoffs = deepcopy(self.points)
            the_last_round_winner = deepcopy(self.last_round_winner_idx)
            self.history.append(
                (the_round, the_players, the_dealer, the_round_counter, the_trick_history, the_playoffs,
                 the_last_round_winner, the_trick_winner_card_history))

        # playing of a single step
        self.round.proceed_round(self.players, action)

        '''
        if the round is over:
            1. save the trick in history
            3. get the winner
            2. update the payoffs
            4. start a new round
            5. count up the round
        '''

        if self.round.is_over:
            self.trick_history.append(cards2list(self.round.trick.copy()))
            self.last_round_winner_idx = self.round.winner_idx
            self.points = self.judger.receive_points(
                self.points, self.round.points)
            self.round_counter += 1
            self.round.start_new_round(self.last_round_winner_idx,
                                       self.num_players, self.top_card, self.trump_color)

        player_id = self.round.current_player_idx
        state = self.get_state(player_id)

        return state, player_id

    def step_back(self) -> bool:
        if len(self.history) > 0:
            self.round, self.players, self.dealer, \
            self.round_counter, self.trick_history, self.points, \
            self.last_round_winner_idx, self.trick_winner_card_history = self.history.pop()
            return True
        return False

    def get_num_players(self) -> int:
        return self.num_players

    @staticmethod
    def get_num_actions() -> int:
        return WizardGame.num_actions

    def get_player_id(self) -> int:
        return self.round.current_player_idx

    def is_over(self) -> bool:
        return self.round_counter > self.max_num_rounds

    def get_payoffs(self) -> list:
        forecasts = [
            self.players[i].forecast for i in range(len(self.players))]

        return self.judger.judge_game_var2(self.points, forecasts)

    def get_legal_actions(self) -> list:
        return self.round.get_legal_actions(self.players[self.round.current_player_idx])
