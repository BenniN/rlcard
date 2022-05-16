from rlcard.games.cego.utils import cards2value


class CegoJudger:
    ''' The class to judge the winner of a round and the points of each player

    instance attributes:
        - np_random: numpy random state
    '''

    def __init__(self, np_random):
        self.np_random = np_random

    def receive_points(self, points, players, player_id, cards) -> list:
        new_points = cards2value(cards)

        # if player is cego player, only he receives the reward
        if players[player_id].is_single_player:
            points[player_id] += new_points

        # if player is not cego player, all players but the cego player receive the reward
        else:
            for i in range(len(players)):
                if not players[i].is_single_player:
                    points[i] += new_points

        return points

    def judge_game(self, points) -> list:
        if points[0] > points[1]:
            return [1, 0, 0, 0]
        else:
            return [0, 1, 1, 1]

    def judge_game_var2(self, points) -> list:
        if points[0] > points[1]:
            return [1, -1, -1, -1]
        else:
            return [-1, 1, 1, 1]