class WizardPlayer:
    ''' Represents a player

    Instance attributes:
        - np_random (numpy.random.RandomState): numpy random state
        - player_id (int): The id of the player
        - is_wizard_player (bool): Whether the player is the wizard player
        - hand (list): The cards in the hand
        - valued_cards (list): The cards layed asside and converted to points
    '''

    def __init__(self, player_id, np_random, is_single_player=False):
        ''' Initilize a player

        Parameters:
            - player_id (int): The id of the player
            - np_random (numpy.random.RandomState): numpy random state
            - is_wizard_player (bool): Whether the player is the wizard player
        '''
        self.np_random = np_random
        self.player_id: int = player_id
        self.is_single_player: bool = is_single_player
        self.hand: list = []
        self.num_tricks= 0

    def get_player_id(self) -> int:
        ''' Return the id of the player
        '''

        return self.player_id
