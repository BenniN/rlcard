from termcolor import colored
import random


def map_suit_to_color(suit) -> str:
    ''' Map the suit to a color

    Parameters:
        - suit: the suit of the card
    '''

    switcher = {
        "r": "red",
        "g": "green",
        "b": "blue",
        "y": "yellow",
        "n": "white",
        "w": "white"
    }
    return switcher.get(suit, "Invalid suit")


class WizardCard:
    ''' A class for a card in Wizard Cardgame

    Class Attributes:
        - info (dict): The information of the card
            - suits (list): possible suits for a card
            - ranks (list): possible ranks for a card
            - cards (list): A list of all the cards
            - red_cards_ranks (list): The ranking of red cards
            - green_cards_ranks (list): The ranking of green cards
            - blue_cards_ranks (list): The ranking of blue cards
            - yellow_cards_ranks (list): The ranking of yellow cards
            - non_colour_card_ranks (list): The ranking of "Narren" and "Wizards"
            - narr_cards_ranks (list): The ranking of Narren
            - wizard_cards_ranks (list): The ranking of Wizards

    Instance Attributes:
        - suit (str): The suit of the card
        - rank (str): The rank of the card
    '''

    info = {
        "suits": ["r", "g", "b", "y", "n", "w", "trump_color"],
        "ranks": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                  "11", "12", "13", "14"],
        "cards": [
            "1-r", "2-r", "3-r", "4-r", "5-r", "6-r", "7-r", "8-r", "9-r", "10-r", "11-r", "12-r", "13-r",
            "1-g", "2-g", "3-g", "4-g", "5-g", "6-g", "7-g", "8-g", "9-g", "10-g", "11-g", "12-g", "13-g",
            "1-b", "2-b", "3-b", "4-b", "5-b", "6-b", "7-b", "8-b", "9-b", "10-b", "11-b", "12-b", "13-b",
            "1-y", "2-y", "3-y", "4-y", "5-y", "6-y", "7-y", "8-y", "9-y", "10-y", "11-y", "12-y", "13-y",
            "0-n", "1-n", "2-n", "3-n", "0-w", "1-w", "2-w", "3-w"
        ],
        "non_colour_card_values": {
            "n": 0,
            "w": 14,
        },
        "red_card_ranks": [],
        "green_cards_ranks": [],
        "blue_cards_ranks": [],
        "yellow_cards_ranks": [],
        "non_colour_card_ranks": [],
        "narr_card_ranks": [],
        "wizard_card_ranks": [],
    }

    def __init__(self, suit, rank, trump_colour=""):
        ''' Initialize the class of WizardCard

        Parameters:
            - suit (str): The suit of card
            - trait (str): The trait of card
        '''
        self.suit = suit
        self.rank = rank
        self.trump_colour = trump_colour

    ''' Comparison function for sorting cards 
        Start
    '''

    def __str__(self) -> str:
        return self.rank + '-' + self.suit

    def __lt__(self, other) -> bool:
        return WizardCard.compare_card_rank(self, other) < 0

    def __gt__(self, other) -> bool:
        return WizardCard.compare_card_rank(self, other) > 0

    # def __eq__(self, other) -> bool:
    #     return WizardCard.compare_card_rank(self, other) == 0

    def __le__(self, other) -> bool:
        return WizardCard.compare_card_rank(self, other) <= 0

    def __ge__(self, other) -> bool:
        return WizardCard.compare_card_rank(self, other) >= 0

    # def __ne__(self, other) -> bool:
    #     return WizardCard.compare_card_rank(self, other) != 0

    def get_value(self) -> float:
        if self.rank in WizardCard.info["ranks"]:
            return WizardCard.info["ranks"][self.rank]

        if self.rank in WizardCard.info["non_colour_card_ranks"]:
            return WizardCard.info["non_colour_card_ranks"][self.rank]

        return 13

    ''' Comparison function for sorting cards 
        End
    '''

    @staticmethod
    def compare_card_rank(card1, card2) -> int:
        ''' Compare the rank of two cards

        Parameters:
            - card1 (WizardCard): The first card
            - card2 (WizardCard): The second card
        '''
        if card1.suit == "w" and card2.suit != "w":
            return 1
        if card1.suit != "w" and card2.suit == "w":
            return -1
        if card1.suit == "n" and card2.suit != "n":
            return -1
        if card1.suit != "n" and card2.suit == "n":
            return 1
        if card1.suit == "n" and card2.suit == "n":
            return 1
        if card1.suit == "w" and card2.suit == "w":
            # first wizard wins so card1.
            return 1

        if card1.suit == "trump_colour" and card2.suit != "trump_colour":
            return 1
        elif card1.suit != "trump_colour" and card2.suit == "trump_colour":
            return -1
        elif card1.suit == "trump_colour" and card2.suit == "trump_colour":
            return WizardCard.info["trump_card_ranks"].index(card1.rank) - WizardCard.info["trump_card_ranks"].index(
                card2.rank)
        elif card1.suit != "trump_colour" and card2.suit != "trump_colour":
            return WizardCard.info["colour_ranks"].index(card1.rank) - WizardCard.info["colour_ranks"].index(
                card2.rank)

        idx_card1 = 0
        if card1.suit == "r":
            idx_card1 = WizardCard.info["red_card_ranks"].index(card1.rank)
        if card1.suit == "g":
            idx_card1 = WizardCard.info["green_cards_ranks"].index(card1.rank)
        if card1.suit == "b":
            idx_card1 = WizardCard.info["blue_card_ranks"].index(card1.rank)
        if card1.suit == "y":
            idx_card1 = WizardCard.info["yellow_cards_ranks"].index(card1.rank)
        if card1.suit == "n":
            idx_card1 = WizardCard.info["non_colour_card_ranks"].index(
                card1.rank)
        if card1.suit == "w":
            idx_card1 = WizardCard.info["non_colour_card_ranks"].index(
                card1.rank)

        idx_card2 = 0
        if card2.suit == "r":
            idx_card2 = WizardCard.info["red_card_ranks"].index(card2.rank)
        if card2.suit == "g":
            idx_card2 = WizardCard.info["green_cards_ranks"].index(card2.rank)
        if card2.suit == "b":
            idx_card2 = WizardCard.info["blue_card_ranks"].index(card2.rank)
        if card2.suit == "y":
            idx_card2 = WizardCard.info["yellow_cards_ranks"].index(card2.rank)
        if card2.suit == "n":
            idx_card2 = WizardCard.info["non_colour_card_ranks"].index(
                card2.rank)
        if card2.suit == "w":
            idx_card2 = WizardCard.info["non_colour_card_ranks"].index(
                card2.rank)

        return idx_card1 - idx_card2

    @staticmethod
    def split_ranks() -> tuple[list, list, list, list, list, list]:
        red_card_ranks = WizardCard.info["ranks"][1:13]
        green_card_ranks = WizardCard.info["ranks"][1:13]
        blue_card_ranks = WizardCard.info["ranks"][1:13]
        yellow_card_ranks = WizardCard.info["ranks"][1:13]
        narr_card_ranks = WizardCard.info["ranks"][0]
        wizard_card_ranks = WizardCard.info["ranks"][14]

        return red_card_ranks, green_card_ranks, blue_card_ranks, yellow_card_ranks, narr_card_ranks, wizard_card_ranks

    @staticmethod
    def print_cards(cards) -> None:
        ''' Print a cards list

        Parameters:
            - cards (list): The cards list
        '''
        if isinstance(cards, str):
            cards = [cards]
        for i, card in enumerate(cards):
            rank, suit = card.split('-')

            print(colored(rank + "-" + suit)),

            if i < len(cards) - 1:
                print(', ', end='')

    @staticmethod
    def setup_sorted_suit_ranks() -> None:
        red_card_ranks, green_card_ranks, blue_card_ranks, yellow_card_ranks, narr_card_ranks, wizard_card_ranks = WizardCard.split_ranks()

        WizardCard.info["red_card_ranks"] = red_card_ranks
        WizardCard.info["green_card_ranks"] = green_card_ranks
        WizardCard.info["blue_card_ranks"] = blue_card_ranks
        WizardCard.info["yellow_card_ranks"] = yellow_card_ranks
        WizardCard.info["narr_card_ranks"] = narr_card_ranks
        WizardCard.info["wizard_card_ranks"] = wizard_card_ranks


# setup for WizardCard class
WizardCard.setup_sorted_suit_ranks()
