import numpy as np
from collections import OrderedDict

import rlcard
from rlcard.envs import Env
from rlcard.games.wizard.utils import ACTION_LIST, ACTION_SPACE
from rlcard.games.wizard.utils import cards2list, encode_observation_var1, encode_observation_perfect_information

DEFAULT_GAME_CONFIG = {
    'game_num_players': 4,
    'game_variant': 'standard',
    # 0: judge by points, 1: judge by game, 2: judge by game var2
    'game_judge_by_points': 2,
    'game_activate_heuristic': True,
    'game_with_perfect_information': False,
}


def map_to_Game(variant_name):
    switcher: dict = {
    }

    return switcher.get(variant_name, "Invalid variant name")


class CegoEnv(Env):
    ''' Cego Environment

    Instance Attributes:
        - name (str): the name of the game
        - default_game_config (dict): the default game config
        - game (Game): the game instance
        - state_shape (list): the shape of the state space
        - action_shape (list): the shape of the action space
    '''

    def __init__(self, config):
        ''' Initialize the Cego environment
        '''
        self.name = 'cego'
        self.default_game_config = DEFAULT_GAME_CONFIG

        # select the proper game variant
        variant = config['game_variant'] if 'game_variant' in config else DEFAULT_GAME_CONFIG['game_variant']
        self.game = map_to_Game(variant)()

        super().__init__(config)
        if self.game.with_perfect_information:
            self.state_shape = [[498] for _ in range(self.num_players)]
        else:
            self.state_shape = [[336] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def _extract_state(self, state) -> OrderedDict:
        ''' Extract the observation for each player

        '''
        extracted_state: dict = OrderedDict()
        legal_actions: OrderedDict = self._get_legal_actions()

        is_raeuber_game = self.game.__class__.__name__ == 'GameRaeuber'

        perfect_info_state = self.get_perfect_information()
        if self.game.with_perfect_information:
            extracted_state['obs'] = encode_observation_var1(
                encode_observation_perfect_information(perfect_info_state),
                is_raeuber_game
            )
        else:
            extracted_state['obs'] = encode_observation_var1(
                state,
                is_raeuber_game
            )
        # setup extracted state
        extracted_state['obs'] = encode_observation_var1(state)
        extracted_state['legal_actions'] = legal_actions
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [
            a for a in state['legal_actions']]
        extracted_state['action_record'] = self.action_recorder

        return extracted_state

    def get_payoffs(self):
        ''' Get the payoffs of the players'''

        payoffs = self.game.get_payoffs()
        return np.array(payoffs)

    def _decode_action(self, action_id):
        ''' Decode the action id into the action
        '''

        legal_ids = self._get_legal_actions()

        # if the action is legal, return the action
        if action_id in legal_ids:
            return ACTION_LIST[action_id]

        # else return a random legal action
        return ACTION_LIST[np.random.choice(legal_ids)]

    def _get_legal_actions(self) -> OrderedDict:
        ''' Get the legal actions of the current state'''

        legal_actions = self.game.get_legal_actions()
        legal_ids = {ACTION_SPACE[action]: None for action in legal_actions}
        return OrderedDict(legal_ids)

    def get_perfect_information(self) -> dict:
        ''' Get the perfect information of the current state '''
        state = {}
        state['num_players'] = self.num_players
        state['hand_cards'] = [cards2list(player.hand)
                               for player in self.game.players]
        state['blind_cards='] = self.game.blind_cards
        state['trick'] = cards2list(self.game.round.trick)
        state['played_tricks'] = self.game.trick_history
        state['current_player'] = self.game.round.current_player_idx
        state['legal_actions'] = self.game.round.get_legal_actions(
            self.game.players[state['current_player']]
        )
        state['winner'] = self.game.round.winner_idx
        state['target'] = str(self.game.round.target) if str(
            self.game.round.target) is not None else None
        state['winner_card'] = str(self.game.round.winner_card) if str(
            self.game.round.winner_card) is not None else None
        state['start_player'] = self.game.round.starting_player_idx
        return state
