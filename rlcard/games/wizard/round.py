from rlcard.games.wizard.utils import cards2list, compare_trick_winner


class WizardRound:
    ''' A Round in Wizard equals one trick

    Class Attributes:
        - num_players: number of players in the game

    Instance Attributes:
        - np_random (numpy.random.RandomState): numpy random state
        - tump_color: the trump color of the round
        - top_card: the top_card decideing the trumpcolor
        - current_player_idx (int): index of the current player
        - starting_player_idx (int): index of the starting player
        - trick (list): the cards in the trick
        - winner_card (Card): the card that currently wins the trick
        - target (Card): the target be served within the trick
        - is_over (bool): whether the round is over
        - winner_idx (int): the index of the player that wins the round
    '''

    num_players: int = 6

    def __init__(self, np_random):
        self.np_random = np_random

    def start_new_round(self, starting_player_idx, num_players, top_card, trump_color) -> None:
        self.trump_color = trump_color
        self.top_card = top_card
        self.num_players = num_players
        self.points = [0 for _ in range(self.num_players)]
        self.judgepoints = [0 for _ in range(self.num_players)]

        self.current_player_idx: int = starting_player_idx
        self.starting_player_idx: int = starting_player_idx
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
            current_winner = compare_trick_winner(
                self.winner_card, card, self.top_card, self.trump_color)
            if current_winner < 0:
                self.winner_card = card
                self.winner_idx = self.current_player_idx

        self.trick.append(card)

        if (len(self.trick)) == self.num_players:
            self.points[self.winner_idx] += 1
            self.is_over = True

        self.current_player_idx = (
                                          self.current_player_idx + 1) % self.num_players

    def get_legal_actions(self, player) -> list:
        ''' get legal actions for current player

        Parameters:
            - player (Player): the current player
        '''

        hand = player.hand  # get hand of current player
        legal_actions = []

        # if no card has been played, all cards are legal

        if self.target is None:
            return cards2list(hand)

        # if the cards fit the suit, they must be played
        for card in hand:
            if card.suit == self.target.suit:
                legal_actions.append(str(card))

        if len(legal_actions) == 0:
            return cards2list(hand)

        for card in hand:
            if card.suit == 'n' or card.suit == 'w':
                legal_actions.append(str(card))

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
        state['round_color'] = self.trump_color
        state['target_card'] = str(self.target) if str(
            self.target) is not None else None
        state['winner_card'] = str(self.winner_card) if str(
            self.winner_card) is not None else None
        state['winner'] = self.winner_idx
        state['legal_actions'] = self.get_legal_actions(player)
        state['start_player'] = self.starting_player_idx
        return state
