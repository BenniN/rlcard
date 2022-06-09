from rlcard.games.cego.utils import cards2value
from rlcard.games.cego import Judger


class CegoJudgerBettel(Judger):
    ''' The standard class to judge the winner of a round and the points of each player

    inherits from:
        - Judger
    '''

    def __init__(self, np_random):
        self.np_random = np_random

    def update_points(self, points, players, winner_player_id, trick_cards) -> list:
        # trick are counted within points(list)
        if players[winner_player_id].is_single_player:
            points[winner_player_id] += 1

        else:
            for i in range(len(players)):
                if not players[i].is_single_player:
                    points[i] += 1

        return points

    def judge_game_zero_to_one(self, points) -> list:
        if points[0] > 0:
            return [0, 1, 1, 1]
        else:
            return [1, 0, 0, 0]

    def judge_game_minusone_to_one(self, points) -> list:
        if points[0]> 0:
            return [-1, 1, 1, 1]
        else: 
            return [1, -1, -1, -1]
