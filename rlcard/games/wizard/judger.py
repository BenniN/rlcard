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

    def receive_judge_points(self, points, last_round_points, player_forecast) -> list:
        judge_points = [0 for _ in range(len(points))]
        # print("forecast", player_forecasts)

        for i in range(len(points)):
            if points[i] <= player_forecast[i]:
                judge_points[i] += last_round_points[i]
            else:
                judge_points[i] -= last_round_points[i]
        # print("actual", points)
        return judge_points

    def judge_game_var222(self, points, player_forecasts) -> list:
        result = [0 for _ in range(len(points))]
        #print("forecast", player_forecasts)

        for i in range(len(points)):
            result[i] = player_forecasts[i]

        return result
        #print("actual", points)


        # if points[0] > points[1]:
        #     return [1, 0, 0, 0]
        # else:
        #     return [0, 1, 1, 1]

    def judge_game_var3(self, points) -> list:
        if points[0] > points[1]:
            return [1, -1, -1, -1]
        else:
            return [-1, 1, 1, 1]

    def judge_game_var2(self, points, player_forecasts) -> list:
        result = [0 for _ in range(len(points))]
        #print("forecast", player_forecasts)

        for i in range(len(points)):
            if points[i] == player_forecasts[i]:
                result[i] = points[i] + 2
            else:
                result[i] = - abs(player_forecasts[i] - points[i])
        #print("actual", points)

        return result
    def judge_game_var22222(self, points, player_forecasts) -> list:
        result = [0 for _ in range(len(points))]
        #print("forecast", player_forecasts)

        for i in range(len(points)):
            if points[i] == player_forecasts[i]:
                result[i] += 1
            else:
                result[i] += 0
        #print("actual", points)

        return result
