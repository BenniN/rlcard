import numpy as np
import json
import os
import csv
import matplotlib.pyplot as plt
from collections import OrderedDict

import random

import rlcard

from rlcard.games.wizard.card import WizardCard as Card

# Read required docs
ROOT_PATH = rlcard.__path__[0]

with open(os.path.join(ROOT_PATH, 'games/wizard/jsondata/action_space.json'), 'r') as file:
    ACTION_SPACE = json.load(file, object_pairs_hook=OrderedDict)
    ACTION_LIST = list(ACTION_SPACE.keys())


def init_forecast_dict():
    result = {}

    player_num_folder_list = ['three_players',
                              'four_players', 'five_players', 'six_players']

    forecast_file_dict = [
        'first_position',
        'average_position',
        'average_position_no_trumpcolor',
        'first_position_no_trumpcolor',
        'trumpcolor'
    ]

    for folder in player_num_folder_list:
        result[folder] = {}
        for file in forecast_file_dict:
            with open(os.path.join(ROOT_PATH, 'games/wizard/jsondata/' + folder + '/' + folder + '_' + file + '.json'),
                      'r') as current_file:
                action_space = json.load(
                    current_file, object_pairs_hook=OrderedDict)
                action_list = list(ACTION_SPACE.keys())
            result[folder][file] = {}
            result[folder][file]['action_space'] = action_space

    return result


forecast_dict = init_forecast_dict()


def init_deck() -> list:
    ''' initialize the wizard deck'''

    deck = []
    card_info = Card.info

    for card in card_info["cards"]:
        rank, suit = card.split('-')
        deck.append(Card(suit, rank))

    return deck


def cards2list(cards) -> list:
    ''' convert cards to list of str'''

    cards_list = []

    for card in cards:
        cards_list.append(str(card))

    return cards_list


def cards2value(cards) -> float:
    ''' get the value of a list of cards'''

    value = 0
    for card in cards:
        value += card.get_card_value()

    return value


def get_tricks_played(tricks) -> list:
    ''' get all the cards out of the tricks list'''

    card_idxs = []
    for trick in tricks:
        for card in trick:
            card_idxs.append(ACTION_SPACE[str(card)])

    return card_idxs


def get_known_cards(hand, top_card, tricks_played, current_trick, start_idx=0) -> list:
    ''' get all cards that are already out of the game '''

    known_cards = []
    if hand is not None:
        known_cards.extend(hand)
    if top_card is not None and top_card != 'None':
        known_cards.append(top_card)
    if tricks_played is not None:
        known_cards.extend([card for trick in tricks_played for card in trick])
    if current_trick is not None:
        known_cards.extend(current_trick)
    card_idxs = [start_idx + ACTION_SPACE[card] for card in known_cards]
    return card_idxs


def get_cards_played(tricks_played, current_trick) -> list:
    ''' get all the cards out of the game list'''

    tricks = tricks_played[:]
    tricks.append(current_trick)

    card_idxs = []
    for trick in tricks:
        for card in trick:
            card_idxs.append(ACTION_SPACE[str(card)])

    return card_idxs


def set_wizard_player_deck(player, blind_cards) -> None:
    """
    this function is a helper function for selecting
    the cards from the blind deck for the wizard player

    the selection process is rule based:

        1. sort the blind and hand cards by rank descending
        2. throw away worst ranked card from blind
        3. take best 2 cards from hand
    """

    #  get hand cards
    hand_cards = player.hand

    # sort cards
    sorted_blinds = sorted(blind_cards, reverse=True)
    sorted_hand = sorted(hand_cards, reverse=True)

    # generate new hand
    new_hand = sorted_blinds[:-1] + sorted_hand[0:2]

    # generate throw aways
    throw_away = sorted_hand[2:]
    throw_away.append(sorted_blinds[-1])

    # set player cards
    player.hand = new_hand
    player.valued_cards = throw_away


def set_observation(obs, plane, indexes):
    ''' set observation of a specific plane

    Parameters:
        - obs (dict): the observation
        - plane (int): the plane to be set
        - indexes (list): the indexes to be set
    '''
    for index in indexes:
        obs[plane][index] = 1


def encode_observation_var0(state):
    ''' the shape of this encoding is (383)

    Parameters:
        - state (dict): the state of the game

    Returns:
        - obs (list): the observation

    Observation Representation
        - [0-59] own cards
        - [60-119] top card
        - [120-179] target card
        - [180-239] cards playable by other players
        - [240-299] winner of trick
        - [300-359] cards in trick
        - [360-383] Game Information
            [360-365] Player who started round
            [366-371] Card played position for trick
            [372-377] player who wins current round
            [378-383] color
            [384-385]
            [384-404]
        '''

    obs = np.zeros((384), dtype=int)

    hand_cards_idx = [ACTION_SPACE[card] for card in state['hand']]
    obs[hand_cards_idx] = 1

    top_card_id = None
    if state['top_card'] != 'None':
        top_card_id = 60 + ACTION_SPACE[state['top_card']]
        obs[top_card_id] = 1

    target_card_id = None
    if state['target_card'] != 'None':
        target_card_id = 120 + ACTION_SPACE[state['target_card']]
        obs[target_card_id] = 1

    known_card_ids = get_known_cards(
        state['hand'], state['top_card'], state['tricks_played'], state['current_trick'], 180)
    obs[range(180, 240)] = 1
    obs[known_card_ids] = 0

    winner_card_id = None
    if state['winner_card'] != 'None':
        winner_card_id = 240 + ACTION_SPACE[state['winner_card']]
        obs[winner_card_id] = 1

    current_trick_card_ids = [300 + ACTION_SPACE[card]
                              for card in state['current_trick']]
    obs[current_trick_card_ids] = 1

    encode_obs_game_info(state, obs, 360)

    return obs


def encode_observation_var1(state):
    ''' the shape of this encoding is (370)

    Parameters:
        - state (dict): the state of the game

    Returns:
        - obs (list): the observation

    Observation Representation
        - [0] predicted amount of cards to win
        - [1-60] own cards
        - [61-120] top card
        - [121-180] target card
        - [181-240] cards playable by other players
        - [241-300] winner of trick
        - [301-360] cards in trick
        - [361-369] Game Information
            [361] Player who started round
            [362] Card played position for trick
            [363] player who wins current round
            [364-369] color
        '''

    obs = np.zeros((370), dtype=int)

    obs[0] = state['current_player_forecast']
    hand_cards_idx = [1 + ACTION_SPACE[card] for card in state['hand']]
    obs[hand_cards_idx] = 1

    top_card_id = None
    if state['top_card'] != 'None':
        top_card_id = 61 + ACTION_SPACE[state['top_card']]
        obs[top_card_id] = 1

    target_card_id = None
    if state['target_card'] != 'None':
        target_card_id = 121 + ACTION_SPACE[state['target_card']]
        obs[target_card_id] = 1

    known_card_ids = get_known_cards(
        state['hand'], state['top_card'], state['tricks_played'], state['current_trick'], 181)
    obs[range(181, 241)] = 1
    obs[known_card_ids] = 0

    winner_card_id = None
    if state['winner_card'] != 'None':
        winner_card_id = 241 + ACTION_SPACE[state['winner_card']]
        obs[winner_card_id] = 1

    current_trick_card_ids = [301 + ACTION_SPACE[card]
                              for card in state['current_trick']]
    obs[current_trick_card_ids] = 1

    encode_obs_game_info_forecast(state, obs, 361)

    return obs


def encode_observation_perfect_information(state):
    ''' the shape of this encoding is (498)
    Parameters:
        - state (dict): the state of the game
    Returns:
        - obs (list): the observation
    Observation Representation
        - [0-59] cards player 0
        - [60-119] cards player 1
        - [120-179] cards player 2
        - [180-239] cards player 3
        - [240-299] cards player 4
        - [300-359] cards player 5
        - [360-419] target card
        - [420-479] cards playable by other players
        - [480-539] winner of trick
        - [540-599] cards in trick
        - [600-623] Game Information
            - [600-605] Player who started round
            - [606-611] Card played position for trick
            - [612-617] player who wins round
            - [618-623] color

    '''
    obs = np.zeros((623), dtype=int)

    for i in range(len(state['hand_cards'])):
        hand_cards_idx = [i * 60 + ACTION_SPACE[card]
                          for card in state['hand_cards'][i]]
        obs[hand_cards_idx] = 1

    target_card_id = None
    if state['target_card'] != 'None':
        target_card_id = 360 + ACTION_SPACE[state['target_card']]
        obs[target_card_id] = 1

    known_card_ids = get_known_cards(
        state['hand'], state['top_card'], state['tricks_played'], state['current_trick'], 420)
    obs[range(420, 480)] = 1
    obs[known_card_ids] = 0

    if state['winner_card'] != 'None':
        winner_card_id = 480 + ACTION_SPACE[state['winner_card']]
        obs[winner_card_id] = 1

    current_trick_card_ids = [520 + ACTION_SPACE[card]
                              for card in state['current_trick']]
    obs[current_trick_card_ids] = 1

    encode_obs_game_info(state, obs, 600)

    return obs


def get_hand_forecast_value(anticipate_max_param, hand, num_players, num_round, top_card, trump_color,
                            current_position):
    hand_value = 0
    for card in hand:
        current_value = get_card_forecast_value(
            anticipate_max_param, card, num_players, num_round, top_card, trump_color, current_position)
        hand_value += current_value
        # print(str(card), current_value)

    # print("hand_value", hand_value)
    return hand_value


def card_rank_to_value(card, action_space):
    if card.suit == "n":
        card_value = action_space["14"] / 100
    elif card.suit == "w":
        card_value = action_space["15"] / 100
    else:
        card_value = action_space[str(card.rank)] / 100
    return card_value


def get_card_forecast_value(anticipate_max_param, card, num_players, num_round, top_card, trump_color,
                            current_position) -> float:
    """
        anticipate_max_param: 0 <= x <= 1
            x=1 : weighting towards extrem(max) postion.
    """

    if num_players == 3:
        path = "three_players"
    elif num_players == 4:
        path = "four_players"
    elif num_players == 5:
        path = "five_players"
    elif num_players == 6:
        path = "six_players"

    card_value = 0
    weighted_value = 0
    pos_1_one_weight = 0
    pos_others_weight = 0

    if num_round < 2:
        '''in the first round the values are fix due to the position of the player
        '''
        if top_card.suit == "n":
            if current_position == 0:

                action_space = forecast_dict[path]['first_position_no_trumpcolor']['action_space']
                card_value = card_rank_to_value(card, action_space)

                return card_value
            else:
                ''' get value through card rank to key for json file second to num_players and three
                players top card narr
                '''

                action_space = forecast_dict[path]['average_position_no_trumpcolor']['action_space']
                card_value = card_rank_to_value(card, action_space)

                return card_value

        # top_card.suit is "w" and card.suit is not "trump_color":
        elif card.suit != trump_color:
            if current_position == 0:
                ''' get value through card rank to key for json file first position and num_players top
                wizard
                '''
                action_space = forecast_dict[path]['first_position']['action_space']
                card_value = card_rank_to_value(card, action_space)

                return card_value
            else:
                ''' get value through card rank to key for json file second to third position and num_players
                top card wizard
                '''

                action_space = forecast_dict[path]['average_position']['action_space']
                card_value = card_rank_to_value(card, action_space)

                return card_value

        elif card.suit == trump_color:
            ''' get value through card rank to key for json file trump color and num_players top card trump
            color
            '''
            action_space = forecast_dict[path]['trumpcolor']['action_space']
            card_value = card_rank_to_value(card, action_space)

            return card_value

    else:
        if top_card is None or top_card.suit == "n":
            action_space_max = forecast_dict[path]['first_position_no_trumpcolor']['action_space']
            action_space_min = forecast_dict[path]['average_position_no_trumpcolor']['action_space']

            pos_1_one_weight = (1 / num_players)
            pos_others_weight = ((num_players - 1) / num_players)
            weighted_value = (
                        anticipate_max_param * pos_1_one_weight + ((1 - anticipate_max_param) * pos_others_weight))
            card_value = ((card_rank_to_value(card, action_space_max) * pos_1_one_weight * anticipate_max_param) + (
                        card_rank_to_value(card, action_space_min) * pos_others_weight * (
                            1 - anticipate_max_param))) / weighted_value

            # card_value = ((card_rank_to_value(card, action_space_max) * anticipate_max_param) + card_rank_to_value(card, action_space_min) * (1 - anticipate_max_param))

            # card_value = (card_rank_to_value(card, action_space_max) + (card_rank_to_value(card, action_space_min) * (num_players - 1))) / num_players
            # card_value = card_rank_to_value(card, action_space_max)
            # card_value = card_rank_to_value(card, action_space_min)

            return card_value

        # top_card.suit is "w" and card.suit is not "trump_color":
        elif card.suit != trump_color:
            action_space_max = forecast_dict[path]['first_position']['action_space']
            action_space_min = forecast_dict[path]['average_position']['action_space']

            # card_value = (card_rank_to_value(card, action_space_max) * anticipate_max_param + (card_rank_to_value(card, action_space_min) * (num_players - 1)) * (1 - anticipate_max_param)) / num_players
            card_value = (card_rank_to_value(card, action_space_max) + (
                    card_rank_to_value(card, action_space_min) * (num_players - 1))) / num_players
            return card_value

        elif card.suit == trump_color:
            ''' get value through card rank to key for json file trump color and num_players top card trump
            color
            '''

            action_space = forecast_dict[path]['trumpcolor']['action_space']

            card_value = card_rank_to_value(card, action_space)

            return card_value


def map_color_to_index(color) -> str:
    ''' Map the suit to a color

    Parameters:
        - suit: the suit of the card
    '''

    switcher = {
        "r": 0,
        "g": 1,
        "b": 2,
        "y": 3,
        "n": 4,
        "w": 5
    }
    return switcher.get(color, "Invalid suit")


def encode_obs_game_info(state, obs, start_idx):
    winner_idx = state['winner']
    start_player_idx = state['start_player']
    current_player_idx = state['current_player']
    round_color = map_color_to_index(state['round_color'])

    obs[start_idx + start_player_idx] = 1

    if winner_idx is not None:
        obs[start_idx + winner_idx] = 1

    if current_player_idx is not None:
        obs[start_idx + 6 + current_player_idx] = 1

    if start_player_idx is not None:
        obs[start_idx + 12 + start_player_idx] = 1

    if round_color is not None and round_color != 'Invalid suit':
        obs[start_idx + 18 + round_color] = 1


def encode_obs_game_info_forecast(state, obs, start_idx):
    winner_idx = state['winner']
    start_player_idx = state['start_player']
    current_player_idx = state['current_player']
    round_color = map_color_to_index(state['round_color'])

    obs[start_idx] = start_player_idx

    if winner_idx is not None:
        obs[start_idx + 1] = winner_idx

    if current_player_idx is not None:
        obs[start_idx + 2] = current_player_idx

    if round_color is not None and round_color != 'Invalid suit':
        obs[start_idx + 3 + round_color] = 1


def save_args_params(args):
    if not os.path.exists(args["log_dir"]):
        os.makedirs(args["log_dir"])

    with open(args["log_dir"] + '/model_params.txt', 'w') as f:
        for key, value in args.items():
            f.write("{}: {}\n".format(key, value))


def create_wizard_dmc_graph(model_path):
    file = open(model_path + '/dmc/logs.csv')

    csvreader = csv.DictReader(file)

    y = []
    x_wizard = []
    x_other = []
    tick = 0

    for row in csvreader:
        if isfloat(row['mean_episode_return_1']):
            x_wizard.append(float(row['mean_episode_return_0']))
            y.append(tick)
            tick += 1
        if isfloat(row['mean_episode_return_1']):
            x_other.append(float(row['mean_episode_return_1']))

    fig, ax = plt.subplots()
    ax.plot(y, x_wizard, label='Wizard Player')
    ax.plot(y, x_other, label='Other Players')
    ax.set(xlabel='Tick', ylabel='reward')
    ax.legend()
    ax.grid()

    fig.savefig(model_path + '/fig.png')


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def get_random_search_args(args):
    res = {}

    for val in args:
        res[val] = random.choice(args[val])

    return res


def args_to_str(args):
    res = ''

    for val in args:
        res += '{}_{}_'.format(val, args[val])

    return res


def load_model(model_path, env=None, position=None, device=None):
    import torch
    if os.path.isfile(model_path):  # Torch model
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)

    return agent


def compare_trick_winner(winner_card, compare_to_card, top_card, trump_color) -> int:
    ''' Compare the winner of a trick

    Parameters:
        - target (WizardCard): The current winner of the trick
        - compare_to_card (WizardCard): The card to compare to
    '''
    trump_color = None
    ignore_trump_color = False

    if top_card == None:
        ignore_trump_color = True

    else:
        if trump_color == "n":
            ignore_trump_color = True

    if (winner_card.suit == "w" and compare_to_card.suit != "w") or (
            winner_card.suit == "w" and compare_to_card.suit == "w"):
        return 1
    if winner_card.suit != "w" and compare_to_card.suit == "w":
        return -1
    if winner_card.suit == "n" and compare_to_card.suit != "n":
        return -1
    if winner_card.suit != "n" and compare_to_card.suit == "n":
        return 1
    if winner_card.suit == "n" and compare_to_card.suit == "n":
        return 1

    if not ignore_trump_color and winner_card.suit != trump_color and compare_to_card.suit == trump_color:
        return -1
    if not ignore_trump_color and winner_card.suit == trump_color and compare_to_card.suit != trump_color:
        return 1
    if not ignore_trump_color and winner_card.suit == trump_color and compare_to_card.suit == trump_color:
        return int(winner_card.rank) - int(compare_to_card.rank)
    if winner_card.suit != trump_color and compare_to_card.suit != trump_color:
        if winner_card.suit == compare_to_card.suit:
            return int(winner_card.rank) - int(compare_to_card.rank)
        else:
            return 1
