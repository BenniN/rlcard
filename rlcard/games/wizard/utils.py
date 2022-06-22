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


def encode_observation_var0(state):
    ''' the shape of this encoding is (336)

    Parameters:
        - state (dict): the state of the game

    Returns:
        - obs (list): the observation

    Observation Representation
        - [0-59] own cards
        - [60-119] predicted trick winning cards
        - [120-179] cards playable by other players
        - [180-239] winner of trick
        - [240-299] cards in trick
        - [300-?] Game Information
            [300-?] Player who started round
            [306-?] Card played position for trick
            [312-?] player who wins current round
        '''

    obs = np.zeros((318), dtype=int)

    return obs

def encode_observation_perfect_information(state, is_raeuber=False):
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
        - [360-419] winner of trick
        - [420-479]
        - ....

    '''
    obs = np.zeros((1000), dtype=int)


    return obs


def encode_obs_game_info(state, obs, start_idx):
    winner_idx = state['winner']
    start_player_idx = state['start_player']
    current_player_idx = state['current_player']

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




