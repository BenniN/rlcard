from rlcard.games.wizard.utils import cards2value


class WizardJudger:
    ''' The class to judge the winner of a round and the points of each player

    instance attributes:
        - np_random: numpy random state
    '''

    def __init__(self, np_random):
        self.np_random = np_random

    def receive_points(self, points, last_round_points) -> list:
        for i in range(len(points)):
            points[i] += last_round_points[i]

        return points

    def judge_game(self, points) -> list:
        if points[0] > points[1]:
            return [1, 0, 0, 0]
        else:
            return [0, 1, 1, 1]

    def judge_game_var3(self, points) -> list:
        if points[0] > points[1]:
            return [1, -1, -1, -1]
        else:
            return [-1, 1, 1, 1]


    def judge_game_var2(self, points, player_forecasts) -> list:
        result = [0 for _ in range(len(points))]

        for i in range(len(points)):
            if points[i] == player_forecasts[i]:
                result[i] = points[i]+2
            else:
                result[i] = - abs(player_forecasts[i]-points[i])
        print("tricks", points)

        return result