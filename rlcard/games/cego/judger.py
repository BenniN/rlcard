from rlcard.games.cego.utils import cards2value
from abc import ABC

class CegoJudger(ABC):
    ''' The abstract class to judge the winner of a round and the points of each player

    instance attributes:
        - np_random: numpy random state

    methods to implement:
        - judge_game_points
        - judge_game_zero_to_one
        - judge_game_minusone_to_one
    '''

    def __init__(self, np_random):
        self.np_random = np_random

    def update_points(self, points, players, winner_player_id, trick_cards) -> list:
        ''' update the points of each player
        
        Args:
            - points (list[int]): the current points of each player
            - players (list[Player]): the players
            - winner_player_id (int): the id of the winner player
            - trick_cards (list[list[int]]): the cards in the current trick

        Returns:
            - game points (list[int]): the new points of each player
        '''
        raise NotImplementedError

    def judge_game_zero_to_one(self, points) -> list:
        ''' judge the game with zero to one reward
        
        Args:
            - points (list[int]): the current points of each player

        Returns:
            - game points (list[int]): the final game points of each player
        '''
        raise NotImplementedError

    def judge_game_minusone_to_one(self, points) -> list:
        ''' judge the game with minus one to one reward
        
        Args:
            - points (list[int]): the current points of each player

        Returns:
            - game points (list[int]): the final game points of each player
        '''
        raise NotImplementedError