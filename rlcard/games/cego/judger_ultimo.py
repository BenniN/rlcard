from rlcard.games.cego.utils import cards2value
from rlcard.games.cego import Judger


class CegoJudgerUltimo(Judger):
    ''' The standard class to judge the winner of a round and the points of each player

    inherits from:
        - Judger
    '''

    def __init__(self, np_random):
        self.np_random = np_random
        self.round_counter = 0

    def update_points(self, points, players, winner_player_id, trick_cards, winner_card= None) -> list:
        self.round_counter += 1
        if self.round_counter == 11 and winner_player_id == 0 and str(winner_card) == '1-trump':
            points[0] += 1

        return points

    def judge_game_zero_to_one(self, points) -> list:
        if points[0] == 1:
            return [1, 0, 0, 0]
        else:
            return [0, 1, 1, 1]

    def judge_game_minusone_to_one(self, points) -> list:
        if points[0] == 1:
            return [1, -1, -1, -1]
        else:
            return [-1, 1, 1, 1]
