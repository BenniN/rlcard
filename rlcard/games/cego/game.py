from copy import deepcopy
import numpy as np
from typing import Any

from abc import ABC
from rlcard.games.cego.utils import cards2list
from rlcard.games.cego import Dealer
from rlcard.games.cego import Judger
from rlcard.games.cego import Round


class CegoGame(ABC):
    ''' The abstract class for a cego game

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

    num_rounds: int = 11
    num_actions: int = 54

    def __init__(self, allow_step_back=False, activate_heuristic=False):
        self.allow_step_back: bool = allow_step_back
        self.np_random: np.random.RandomState = np.random.RandomState()
        self.num_players: int = 4  # there are always 4 players in this game
        self.points: list[int] = [0 for _ in range(self.num_players)]

        self.activate_heuristic: bool = activate_heuristic
        self.dealer: Dealer = None
        self.players: list = None
        self.judger: Judger = None
        self.round: Round = None
        self.round_counter: int = None
        self.history: list = None
        self.trick_history: list = None
        self.blind_cards: list = None
        self.last_round_winner_idx: int = None
        self.judge_by_points: int = 0

    def configure(self, game_config) -> None:
        ''' Specify some game specific parameters, such as number of players '''

        self.num_players = game_config['game_num_players']
        self.activate_heuristic = game_config['game_activate_heuristic']
        self.judge_by_points = game_config['game_judge_by_points']
        self.with_perfect_information = game_config['game_with_perfect_information']

    def init_game(self) -> tuple[dict, Any]:
        raise NotImplementedError

    def get_state(self, player_id) -> dict:
        ''' get current state of the game 

        Parameters:
            - player_id: the id of the player

        '''

        state = self.round.get_state(self.players[player_id])
        state['num_players'] = self.get_num_players()
        state['current_player'] = self.round.current_player_idx
        state['current_trick_round'] = self.round_counter
        state['played_tricks'] = self.trick_history
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
            the_playoffs = deepcopy(self.points)
            the_last_round_winner = deepcopy(self.last_round_winner_idx)
            self.history.append(
                (the_round, the_players, the_dealer, the_round_counter, the_trick_history, the_playoffs, the_last_round_winner))

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
            self.on_round_over()

        player_id = self.round.current_player_idx
        state = self.get_state(player_id)

        return state, player_id

    def on_round_over(self) -> None:
        self.trick_history.append(cards2list(self.round.trick.copy()))
        self.last_round_winner_idx = self.round.winner_idx
        self.points = self.judger.update_points(
            self.points,
            self.players,
            self.last_round_winner_idx,
            self.round.trick.copy()
        )
        self.round.start_new_round(self.last_round_winner_idx)
        self.round_counter += 1

    def step_back(self) -> bool:
        if len(self.history) > 0:
            self.round, self.players, self.dealer, \
                self.round_counter, self.trick_history, self.points, self.last_round_winner_idx = self.history.pop()
            return True
        return False

    def get_num_players(self) -> int:
        return self.num_players

    @staticmethod
    def get_num_actions() -> int:
        return CegoGame.num_actions

    def get_player_id(self) -> int:
        return self.round.current_player_idx

    def is_over(self) -> bool:
        return self.round_counter >= CegoGame.num_rounds

    def get_payoffs(self) -> list:
        if self.judge_by_points == 0:
            return self.points
        if self.judge_by_points == 1:
            return self.judger.judge_game_zero_to_one(self.points)
        return self.judger.judge_game_minusone_to_one(self.points)

    def get_legal_actions(self) -> list:
        return self.round.get_legal_actions(self.players[self.round.current_player_idx])
