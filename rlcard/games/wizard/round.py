# from rlcard.games.wizard.card import WizardCard as Card
from rlcard.games.wizard.utils import cards2list, compare_trick_winner


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

    def start_new_round(self, starting_player_idx, num_players, sub_rounds_to_play, top_card) -> None:
        self.top_card = top_card
        self.round_color = None
        self.num_players = num_players
        self.points = [0 for _ in range(self.num_players)]
        print(self.top_card)
        if self.top_card != None:
            rank, suit = str(self.top_card).split('-')
            self.round_color = suit
        self.current_player_idx: int = starting_player_idx
        self.starting_player_idx: int = starting_player_idx
        self.sub_rounds_to_play = sub_rounds_to_play
        self.sub_round_counter = 0
        self.players_that_played_cards: int = starting_player_idx
        self.trick: list = []
        self.winner_card = None
        self.target = None
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
        if self.target == None and card.suit != 'n' and card.suit != 'w':
            self.target = card

        if len(self.trick) == 0:
            self.winner_card = card
            self.winner_idx = self.current_player_idx
        else:
            current_winner = compare_trick_winner(self.winner_card, card, self.top_card)
            if current_winner < 0:
                self.winner_card = card
                self.winner_idx = self.current_player_idx

        self.trick.append(card)

        # if the trick is full, this round is over
        # if len(self.trick) == WizardRound.num_players:
        #     self.is_over = True

        self.players_that_played_cards += 1

        self.current_player_idx = (
                                          self.current_player_idx + 1) % self.num_players

        if self.players_that_played_cards == self.num_players:
            self.points[self.winner_idx] += 1
            self.players_that_played_cards = 0
            self.current_player_idx = self.winner_idx
            self.sub_round_counter += 1

        if self.players_that_played_cards == self.num_players and self.sub_round_counter == self.sub_rounds_to_play:
            self.is_over = True

    def get_legal_actions(self, player) -> list:
        ''' get legal actions for current player

        Parameters:
            - player (Player): the current player
        '''

        hand = player.hand  # get hand of current player
        hand_size = player.hand_size
        legal_actions = []

        # if no card has been played, all cards are legal

        if self.target is None:
            print("in get legal actions target = none: ", legal_actions)
            print("cards2list if target is None", cards2list(hand))
            return cards2list(hand)

        # if the cards fit the suit, they must be played
        for card in hand:
            if card.suit == self.target.suit:
                print("card suit = target suit:", legal_actions.append(str(card)))
                legal_actions.append(str(card))

        if len(legal_actions) == 0:
            print("hand size while legal actions == 0: ", len(hand))
            print("still no legal actions", legal_actions, cards2list(hand))
            return cards2list(hand)
        print("hand should be legal actions:", legal_actions)

        for card in hand:
            if card.suit == 'n' or card.suit == 'w':
                legal_actions.append(str(card))
                print("n or w should be in legal actions", legal_actions)

        print("in legal actions before return", legal_actions)
        return legal_actions

    def get_state(self, player) -> dict:
        ''' get state for current player

        Parameters:
            - player (Player): the current player
        '''

        state = {}
        state['hand'] = cards2list(player.hand)
        state['current_trick'] = cards2list(self.trick)
        state['top_card'] = str(self.top_card)
        state['round_color'] = self.round_color
        state['target_card'] = str(self.target) if str(
            self.target) is not None else None
        state['winner_card'] = str(self.winner_card) if str(
            self.winner_card) is not None else None
        state['winner'] = self.winner_idx
        state['valued_cards'] = cards2list(
            player.valued_cards) if player.is_single_player else []
        state['legal_actions'] = self.get_legal_actions(player)
        state['start_player'] = self.starting_player_idx
        return state

    def get_hand_value(self, player) -> int:
        ''' get handvalue for current player
            the handvalue is needed for the forcasting

                Parameters:
                    - player (Player): the current player
                '''

        hand_value = 0
        hand = player.hand
        handsize = len(player.hand)

        if handsize == 1:
            if self.current_player_idx == 0:
                    for card in hand:
                        cards2list(hand)





        return hand_value

