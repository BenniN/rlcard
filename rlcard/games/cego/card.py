from termcolor import colored


def map_suit_to_color(suit) -> str:
    ''' Map the suit to a color

    Parameters:
        - suit: the suit of the card
    '''

    switcher = {
        "c": "grey",
        "d": "red",
        "h": "red",
        "s": "grey",
        "trump": "yellow",
    }
    return switcher.get(suit, "Invalid suit")


def map_suit_to_symbol(suit) -> str:
    ''' Map the suit to a symbol

    Parameters:
        - suit: the suit of the card
    '''

    switcher = {
        "c": "♣",
        "d": "♦",
        "h": "♥",
        "s": "♠",
        "trump": "★",
    }

    return switcher.get(suit, "Invalid suit")


class CegoCard:
    ''' A class for a card in Cego game

    Class Attributes:
        - info (dict): The information of the card
            - suits (list): possible suits for a card
            - ranks (list): possible ranks for a card
            - cards (list): A list of all the cards
            - color_card_values (dict): The value mapping for color cards
            - trump_card_values (dict): The value mapping for trump cards
            - black_cards_ranks (list): The ranking of black cards
            - red_cards_ranks (list): The ranking of red cards
            - trump_cards_ranks (list): The ranking of trump cards

    Instance Attributes:
        - suit (str): The suit of the card
        - rank (str): The rank of the card
    '''

    info = {
        "suits": ["c", "d", "h", "s", "trump"],
        "ranks": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                  "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                  "21", "gstiess",
                  "b", "r", "d", "k"],
        "cards": [
            "7-c", "8-c", "9-c", "10-c", "b-c", "r-c", "d-c", "k-c",
            "7-s", "8-s", "9-s", "10-s", "b-s", "r-s", "d-s", "k-s",
            "4-d", "3-d", "2-d", "1-d", "b-d", "r-d", "d-d", "k-d",
            "4-h", "3-h", "2-h", "1-h", "b-h", "r-h", "d-h", "k-h",
            "1-trump", "2-trump", "3-trump", "4-trump", "5-trump",
            "6-trump", "7-trump", "8-trump", "9-trump", "10-trump",
            "11-trump", "12-trump", "13-trump", "14-trump", "15-trump",
            "16-trump", "17-trump", "18-trump", "19-trump", "20-trump",
            "21-trump", "gstiess-trump"
        ],
        "color_card_values": {
            "k": 4.5,
            "d": 3.5,
            "r": 2.5,
            "b": 1.5,
        },
        "trump_card_values": {
            "1": 4.5,
            "21": 4.5,
            "gstiess": 4.5,
        },
        "black_cards_ranks": [],
        "red_card_ranks": [],
        "trump_card_ranks": []
    }

    def __init__(self, suit, rank):
        ''' Initialize the class of CegoCard

        Parameters:
            - suit (str): The suit of card
            - trait (str): The trait of card
        '''
        self.suit = suit
        self.rank = rank

    ''' Comparison function for sorting cards 
        Start
    '''

    def __str__(self) -> str:
        return self.rank + '-' + self.suit

    def __lt__(self, other) -> bool:
        return CegoCard.compare_card_rank(self, other) < 0

    def __gt__(self, other) -> bool:
        return CegoCard.compare_card_rank(self, other) > 0

    def __eq__(self, other) -> bool:
        return CegoCard.compare_card_rank(self, other) == 0

    def __le__(self, other) -> bool:
        return CegoCard.compare_card_rank(self, other) <= 0

    def __ge__(self, other) -> bool:
        return CegoCard.compare_card_rank(self, other) >= 0

    def __ne__(self, other) -> bool:
        return CegoCard.compare_card_rank(self, other) != 0

    def get_value(self) -> float:
        if self.rank in CegoCard.info["color_card_values"]:
            return CegoCard.info["color_card_values"][self.rank]

        if self.suit == "trump" and self.rank in CegoCard.info["trump_card_values"]:
            return CegoCard.info["trump_card_values"][self.rank]

        return 0.5

    ''' Comparison function for sorting cards 
        End
    '''

    @staticmethod
    def compare_card_rank(card1, card2) -> int:
        ''' Compare the rank of two cards

        Parameters:
            - card1 (CegoCard): The first card
            - card2 (CegoCard): The second card
        '''

        if card1.suit == "trump" and card2.suit != "trump":
            return 1
        elif card1.suit != "trump" and card2.suit == "trump":
            return -1
        elif card1.suit == "trump" and card2.suit == "trump":
            return CegoCard.info["trump_card_ranks"].index(card1.rank) - CegoCard.info["trump_card_ranks"].index(card2.rank)

        idx_card1 = 0
        if card1.suit == "d" or card1.suit == "h":
            idx_card1 = CegoCard.info["red_card_ranks"].index(card1.rank)
        else:
            idx_card1 = CegoCard.info["black_cards_ranks"].index(card1.rank)

        idx_card2 = 0
        if card2.suit == "d" or card2.suit == "h":
            idx_card2 = CegoCard.info["red_card_ranks"].index(card2.rank)
        else:
            idx_card2 = CegoCard.info["black_cards_ranks"].index(card2.rank)

        return idx_card1 - idx_card2

    @staticmethod
    def compare_trick_winner(target, compare_to_card) -> int:
        ''' Compare the winner of a trick

        Parameters:
            - target (CegoCard): The current winner of the trick
            - compare_to_card (CegoCard): The card to compare to
        '''

        if target.suit == compare_to_card.suit:
            if target.suit == "trump":
                return CegoCard.info["trump_card_ranks"].index(target.rank) - CegoCard.info["trump_card_ranks"].index(compare_to_card.rank)
            elif target.suit == "d" or target.suit == "h":
                return CegoCard.info["red_card_ranks"].index(target.rank) - CegoCard.info["red_card_ranks"].index(compare_to_card.rank)
            else:
                return CegoCard.info["black_cards_ranks"].index(target.rank) - CegoCard.info["black_cards_ranks"].index(compare_to_card.rank)
        elif compare_to_card.suit == "trump":
            return -1
        else:
            return 1

    @staticmethod
    def split_ranks() -> tuple[list, list, list]:
        black_cards_ranks = CegoCard.info["ranks"][6:10] + \
            CegoCard.info["ranks"][22:]
        red_card_ranks = CegoCard.info["ranks"][0:4][::-1] + \
            CegoCard.info["ranks"][22:]
        trump_card_ranks = CegoCard.info["ranks"][:22]

        return black_cards_ranks, red_card_ranks, trump_card_ranks

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

            print(colored(rank+"-" + map_suit_to_symbol(suit),
                          map_suit_to_color(suit)), end="")

            if i < len(cards) - 1:
                print(', ', end='')

    @staticmethod
    def setup_sorted_suit_ranks() -> None:
        black_cards_ranks, red_card_ranks, trump_card_ranks = CegoCard.split_ranks()

        CegoCard.info["black_cards_ranks"] = black_cards_ranks
        CegoCard.info["red_card_ranks"] = red_card_ranks
        CegoCard.info["trump_card_ranks"] = trump_card_ranks


# setup for CegoCard class
CegoCard.setup_sorted_suit_ranks()
