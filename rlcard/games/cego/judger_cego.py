from rlcard.games.cego.utils import cards2value
from rlcard.games.cego import Judger


class CegoJudgerStandard(Judger):
    ''' The standard class to judge the winner of a round and the points of each player

    inherits from:
        - Judger
    '''

    def __init__(self, np_random):
        self.np_random = np_random

    def update_points(self, points, players, winner_player_id, trick_cards) -> list:

        new_points = cards2value(trick_cards)

        # if player is cego player, only he receives the reward
        if players[winner_player_id].is_single_player:
            points[winner_player_id] += new_points

        # if player is not cego player, all players but the cego player receive the reward
        else:
            for i in range(len(players)):
                if not players[i].is_single_player:
                    points[i] += new_points

        return points

    def judge_game_zero_to_one(self, points) -> list:
        if points[0] > points[1]:
            return [1, 0, 0, 0]
        else:
            return [0, 1, 1, 1]

    def judge_game_minusone_to_one(self, points) -> list:
        if points[0] > points[1]:
            return [1, -1, -1, -1]
        else:
            return [-1, 1, 1, 1]