from rlcard.games.wizard.utils import init_deck


class WizardDealer:
    ''' The class to deal the cards to the players

     Class Attributes:
         - num_player_cards: the number of cards each player gets
         - num_round: the number of the current round

     Instance Attributes:
         - np_random: numpy random state
         - deck: the deck of cards
     '''

    ''' Initialize a wizard dealer class
    '''

    def __init__(self, np_random):
        self.np_random = np_random
        self.deck = init_deck()
        self.shuffle()

    def shuffle(self):
        ''' Shuffle the deck
        '''
        self.np_random.shuffle(self.deck)

    def deal_cards(self, player, num):
        for _ in range(num):
            player.hand.append(self.deck.pop())

    def flip_top_card(self):
        ''' Flip top card when a new game starts

        Returns:
            (object): The object of UnoCard at the top of the deck
        '''
        if len(self.deck) == 0:
            return None
        top_card = self.deck.pop()
        return top_card
