from rlcard.games.wizard.card import WizardCard as Card
from rlcard.games.wizard.utils import cards2list


class WizardRound:
    ''' A Round in Wizard equals one trick

    Class Attributes:
        - num_players: number of players in the game

    Instance Attributes:
        - np_random (numpy.random.RandomState): numpy random state
        - current_player_idx (int): index of the current player
        - starting_player_idx (int): index of the starting player
        - trick (list): the cards in the trick
        - winner_card (Card): the card that currently wins the trick
        - target (Card): the target be served within the trick
        - is_over (bool): whether the round is over
        - winner_idx (int): the index of the player that wins the round
    '''

    num_players: int = 4

    def __init__(self, np_random):
        self.np_random = np_random

    def start_new_round(self, starting_player_idx) -> None:
        self.current_player_idx: int = starting_player_idx
        self.starting_player_idx: int = starting_player_idx
        self.trick: list = []
        self.winner_card: Card = None
        self.target: Card = None
        self.is_over: bool = False
        self.winner_idx: int = None

    def proceed_round(self, players, action) -> None:
        ''' keeps the round running

        Parameters:
            - players (list): list of players
            - action (str): the action of the current player
        '''

        """ Logs for Testing """
        # print("current player: ", self.current_player_idx)
        # print("Player Cards: ", cards2list(
        #     players[self.current_player_idx].hand))
        # print("Target Card: ", str(self.target))
        # print("Played Card: ", action)

        # get current player
        player = players[self.current_player_idx]

        # get and remove card from player hand
        remove_index = None
        for index, card in enumerate(player.hand):
            if str(card) == action:
                remove_index = index
                break

        card = player.hand.pop(remove_index)

        # if no card has been player, the first card is the target
        if len(self.trick) == 0:
            self.winner_card = card
            self.winner_idx = self.current_player_idx
        else:
            current_winner = Card.compare_trick_winner(self.winner_card, card)
            if current_winner < 0:
                self.winner_card = card
                self.winner_idx = self.current_player_idx

        self.trick.append(card)

        # if the trick is full, this round is over
        if len(self.trick) == WizardRound.num_players:
            self.is_over = True

        self.current_player_idx = (
            self.current_player_idx + 1) % WizardRound.num_players

    def get_legal_actions(self, player) -> list:
        ''' get legal actions for current player

        Parameters:
            - player (Player): the current player
        '''

        hand = player.hand  # get hand of current player
        legal_actions = []

        # if no card has been player, all cards are legal
        if len(self.trick) == 0:
            legal_actions = cards2list(hand)
            return legal_actions

        self.target = self.trick[0]  # get the first card in trick

        # if the cards fit the suit, they must be played
        for card in hand:
            if card.suit == self.target.suit:
                legal_actions.append(str(card))

        if len(legal_actions) > 0:
            return legal_actions

        # if no card of the same suit is on the hand, a trump card must be played
        for card in hand:
            if card.suit == 'trump':
                legal_actions.append(str(card))

        if len(legal_actions) > 0:
            return legal_actions

        # else, all other cards are legal
        return cards2list(hand)

    def get_state(self, player) -> dict:
        ''' get state for current player

        Parameters:
            - player (Player): the current player
        '''

        state = {}
        state['hand'] = cards2list(player.hand)
        state['trick'] = cards2list(self.trick)
        state['target'] = str(self.target) if str(
            self.target) is not None else None
        state['winner_card'] = str(self.winner_card) if str(
            self.winner_card) is not None else None
        state['winner'] = self.winner_idx
        state['valued_cards'] = cards2list(
            player.valued_cards) if player.is_single_player else []
        state['legal_actions'] = self.get_legal_actions(player)
        state['start_player'] = self.starting_player_idx
        return state
