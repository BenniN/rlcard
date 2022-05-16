import numpy as np
import json
import os
from collections import OrderedDict

import rlcard

from rlcard.games.cego.card import CegoCard as Card

# Read required docs
ROOT_PATH = rlcard.__path__[0]

with open(os.path.join(ROOT_PATH, 'games/cego/jsondata/action_space.json'), 'r') as file:
    ACTION_SPACE = json.load(file, object_pairs_hook=OrderedDict)
    ACTION_LIST = list(ACTION_SPACE.keys())


def init_deck() -> list:
    ''' initialize the cego deck'''

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


def set_cego_player_deck(player, blind_cards) -> None:
    """ 
    this function is a helper function for selecting 
    the cards from the blind deck for the cego player

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

    encode_obs_game_info(state, obs, 216)

    return obs


def encode_observation_var1(state):
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

    encode_obs_game_info(state, obs, 324)

    return obs


def encode_observation_var2(state):
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
    obs[known_cards_idxs] = 0

    known_cards_idxs = get_known_cards(
        state['hand'], state['valued_cards'], state['played_tricks'], state['trick'], 54)

    obs[known_cards_idxs] = 1

    if state['winner_card'] != 'None':
        obs[108 + ACTION_SPACE[state['winner_card']]] = 1

    obs[trick_cards_idx] = 1

    encode_obs_game_info(state, obs, 216)

    return obs


def encode_observation_var3(state):
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
    obs[known_cards_idxs] = 0

    known_cards_idxs = get_known_cards(
        state['hand'], state['valued_cards'], state['played_tricks'], state['trick'], 54)

    obs[known_cards_idxs] = 1

    if state['winner_card'] != 'None':
        obs[108 + ACTION_SPACE[state['winner_card']]] = 1

    obs[trick_cards_idx] = 1

    if len(state["trick"]) > 0:
        obs[216 + ACTION_SPACE[state["trick"][0]]] = 1

    encode_obs_game_info(state, obs, 270)

    return obs


def encode_observation_var4(state):
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

    encode_obs_game_info(state, obs, 270)

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


def valid_cego(cego_player_cards) -> bool:
    ''' This function checks if it would be valid for the 
    for the cego player to play cego.

    Parameters:
        - cego_player_cards (list): the cards of the cego player

    Returns:
        - valid (bool): The base is that the player has
            at least 15 points on his hand.
    '''

    value = cards2value(cego_player_cards)
    return value >= 15
