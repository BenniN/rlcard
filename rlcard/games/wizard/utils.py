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


def init_deck() -> list:
    ''' initialize the wizard deck'''

    deck = []
    card_info = Card.info

    for card in card_info["cards"]:

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
        value += card.get_value()

    return value


def get_tricks_played(tricks) -> list:
    ''' get all the cards out of the tricks list'''

    card_idxs = []
    for trick in tricks:
        for card in trick:
            card_idxs.append(ACTION_SPACE[str(card)])

    return card_idxs


def get_known_cards(hand, valued_cards, tricks_played, current_trick, start_idx=0) -> list:
    ''' get all cards that are already out of the game '''

    known_cards = []
    if hand is not None:
        known_cards.extend(hand)
    if valued_cards is not None:
        known_cards.extend(valued_cards)
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

    # print("get_cards_played:",tricks)

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


def encode_observation_var0(state, is_raeuber=False):
    ''' the shape of this encoding is (228)

    Parameters:
        - state (dict): the state of the game

    Returns:
        - obs (list): the observation

    Observation Representation
        - [0-53] own cards
        - [54-107] cards playable by other players
        - [108-161] winner of trick
        - [162-215] cards in trick
        - [216] Game Information
            - [216-219]: who is part of the team
            - [220-223]: who wins the current round
            - [224-227]: player who started the trick round
    '''
    obs = np.zeros((228), dtype=int)

    hand_cards_idx: list = []
    trick_cards_idx: list = []

    hand_cards_idx = [ACTION_SPACE[card] for card in state['hand']]
    trick_cards_idx = [162 + ACTION_SPACE[card] for card in state['trick']]

    obs[hand_cards_idx] = 1

    known_cards_idxs = get_known_cards(
        state['hand'], state['valued_cards'], state['played_tricks'], state['trick'], 54)

    obs[range(54, 107)] = 1
    # unset all cards that are knowingly out of the game
    obs[known_cards_idxs] = 0

    if state['winner_card'] != 'None':
        obs[108 + ACTION_SPACE[state['winner_card']]] = 1

    obs[trick_cards_idx] = 1

    encode_obs_game_info(state, obs, 216, is_raeuber)

    return obs


def encode_observation_var1(state, is_raeuber=False):
    ''' the shape of this encoding is (336)

    Parameters:
        - state (dict): the state of the game

    Returns:
        - obs (list): the observation

    Observation Representation
        - [0-53] own cards
        - [54-107] cards playable by other players
        - [108-161] winner of trick
        - [162-215] first trick card
        - [216-269] second trick card
        - [270-323] third trick card
        - [324-335] Game Information
            - [324-327]: who is part of the team
            - [328-331]: who wins the current round
            - [332-335]: player who started the trick round
    '''
    obs = np.zeros((336), dtype=int)

    hand_cards_idx: list = []

    hand_cards_idx = [ACTION_SPACE[card] for card in state['hand']]

    obs[hand_cards_idx] = 1

    known_cards_idxs = get_known_cards(
        state['hand'], state['valued_cards'], state['played_tricks'], state['trick'], 54)

    obs[range(54, 107)] = 1
    # unset all cards that are out of the game
    obs[known_cards_idxs] = 0

    if state['winner_card'] != 'None':
        obs[108 + ACTION_SPACE[state['winner_card']]] = 1

    for i in range(len(state["trick"])):
        obs[162 + (i*54) + ACTION_SPACE[state["trick"][i]]] = 1

    encode_obs_game_info(state, obs, 324, is_raeuber)

    return obs


def encode_observation_var2(state, is_raeuber=False):
    ''' the shape of this encoding is (228)

    Parameters:
        - state (dict): the state of the game

    Returns:
        - obs (list): the observation

    Observation Representation
        - [0-53] own cards
        - [54-107] cards already out of the game
        - [108-161] winner of trick
        - [162-215] cards in trick
        - [216] Game Information
            - [216-219]: who is part of the team
            - [220-223]: who wins the current round
            - [224-227]: player who started the trick round
    '''
    obs = np.zeros((228), dtype=int)

    hand_cards_idx: list = []
    trick_cards_idx: list = []

    hand_cards_idx = [ACTION_SPACE[card] for card in state['hand']]
    trick_cards_idx = [162 + ACTION_SPACE[card] for card in state['trick']]

    obs[hand_cards_idx] = 1

    obs[range(54, 107)] = 1
    # unset all cards that are out of the game
    #obs[known_cards_idxs] = 0

    known_cards_idxs = get_known_cards(
        state['hand'], state['valued_cards'], state['played_tricks'], state['trick'], 54)

    obs[known_cards_idxs] = 1

    if state['winner_card'] != 'None':
        obs[108 + ACTION_SPACE[state['winner_card']]] = 1

    obs[trick_cards_idx] = 1

    encode_obs_game_info(state, obs, 216, is_raeuber)

    return obs


def encode_observation_var3(state, is_raeuber=False):
    ''' the shape of this encoding is (282)

    Parameters:
        - state (dict): the state of the game

    Returns:
        - obs (list): the observation

    Observation Representation
        - [0-53] own cards
        - [54-107] cards already out of the game
        - [108-161] winner of trick
        - [162-215] cards in trick
        - [216-269] target card
        - [270] Game Information
            - [270-273]: who is part of the team
            - [274-277]: who wins the current round
            - [278-281]: player who started the trick round
    '''
    obs = np.zeros((228), dtype=int)

    hand_cards_idx: list = []
    trick_cards_idx: list = []

    hand_cards_idx = [ACTION_SPACE[card] for card in state['hand']]
    trick_cards_idx = [162 + ACTION_SPACE[card] for card in state['trick']]

    obs[hand_cards_idx] = 1

    obs[range(54, 107)] = 1
    # unset all cards that are out of the game
    #obs[known_cards_idxs] = 0

    known_cards_idxs = get_known_cards(
        state['hand'], state['valued_cards'], state['played_tricks'], state['trick'], 54)

    obs[known_cards_idxs] = 1

    if state['winner_card'] != 'None':
        obs[108 + ACTION_SPACE[state['winner_card']]] = 1

    obs[trick_cards_idx] = 1

    if len(state["trick"]) > 0:
        obs[216 + ACTION_SPACE[state["trick"][0]]] = 1

    encode_obs_game_info(state, obs, 270, is_raeuber)

    return obs


def encode_observation_var4(state, is_raeuber=False):
    ''' the shape of this encoding is (282)

    Parameters:
            - state (dict): the state of the game

    Returns:
        - obs (list): the observation

    Observation Representation
        - [0-53] own cards
        - [54-107] cards playable by other players
        - [108-161] first trick card
        - [162-215] second trick card
        - [216-269] third trick card
        - [270] Game Information
            - [270-273]: who is part of the team
            - [274-277]: who wins the current round
            - [278-281]: player who started the trick round
    '''
    obs = np.zeros((282), dtype=int)

    hand_cards_idx: list = []

    hand_cards_idx = [ACTION_SPACE[card] for card in state['hand']]

    obs[hand_cards_idx] = 1

    known_cards_idxs = get_known_cards(
        state['hand'], state['valued_cards'], state['played_tricks'], state['trick'], 54)

    obs[range(54, 107)] = 1
    # unset all cards that are out of the game
    obs[known_cards_idxs] = 0

    for i in range(len(state["trick"])):
        obs[108 + (i*54) + ACTION_SPACE[state["trick"][i]]] = 1

    encode_obs_game_info(state, obs, 270, is_raeuber)

    return obs


def encode_observation_perfect_information(state, is_raeuber=False):
    ''' the shape of this encoding is (498)

    Parameters:
        - state (dict): the state of the game

    Returns:
        - obs (list): the observation

    Observation Representation
        - [0-53] cards player 0
        - [54-107] cards player 1
        - [108-161] cards player 2
        - [162-215] cards player 3
        - [216-269] out_of_game_cards
        - [270-323] winner of trick
        - [324-377] first trick card
        - [378-431] second trick card
        - [432-485] third trick card
        - [486-497] Game Information
            - [486-489]: who is part of the team
            - [490-493]: who wins the current round
            - [494-497]: player who started the trick round
    '''
    obs = np.zeros((498), dtype=int)

    for i in range(len(state['hand_cards'])):
        hand_cards_idx = [i*54 + ACTION_SPACE[card]
                          for card in state['hand_cards'][i]]
        obs[hand_cards_idx] = 1

    known_cards_idxs = get_known_cards(
        None, state['valued_cards'], state['played_tricks'], state['trick'], 216)

    obs[known_cards_idxs] = 1

    if state['winner_card'] != 'None':
        obs[270 + ACTION_SPACE[state['winner_card']]] = 1

    for i in range(len(state["trick"])):
        obs[324 + (i*54) + ACTION_SPACE[state["trick"][i]]] = 1

    encode_obs_game_info(state, obs, 486, is_raeuber)

    return obs


def encode_obs_game_info(state, obs, start_idx, is_raeuber=False):
    winner_idx = state['winner']
    start_player_idx = state['start_player']
    current_player_idx = state['current_player']

    if is_raeuber:
        obs[current_player_idx] = 1
    else:
        if current_player_idx == 0:
            obs[start_idx] = 1
        else:
            obs[[start_idx+1, start_idx+2, start_idx+3]] = 1

    if winner_idx != None:
        obs[start_idx+4 + winner_idx] = 1

    if start_player_idx != None:
        obs[start_idx+8+start_player_idx] = 1

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




